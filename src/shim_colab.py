"""
Colab shim: minimal FastAPI server that exposes /v1/chat/completions for an HF model
via transformers.pipeline. Compatible with run_baseline.py --base-url flag.

Two modes:
  1) Real mode: loads the HF model (torch + transformers required, use GPU).
  2) Mock mode: deterministic simulator using the research prompt + verdict parser.
     No torch needed. Use `--mock` flag or set SHIM_MODE=mock env var.

Run locally (mock):
    python src/shim_colab.py --mock
Run on Colab (real, T4 GPU):
    python src/shim_colab.py

IMPORTANT: run_baseline.py appends "/chat/completions" to --base-url, so pass
    --base-url http://localhost:8000/v1
otherwise every request 404s and all predictions are unparsed.

Endpoints:
  GET  /health              -> status
  GET  /v1/models           -> list model
  POST /v1/chat/completions -> OpenAI-compatible chat completions
"""
import os
import sys
import time
from typing import List

# ensure project root (parent of src) is importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.prompting import build_messages, parse_verdict

# ────── configure mock/real mode ──────
USE_MOCK = os.getenv("SHIM_MODE", "").lower() == "mock"
if len(sys.argv) > 1 and sys.argv[1] == "--mock":
    USE_MOCK = True

DEFAULT_MODEL = os.getenv("SHIM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# deterministic mock verdicts, keyed by short model name
MOCK_VERDICTS = {
    ("qwen2.5-3b", "sms_smishing"): 1,     # malicious
    ("qwen2.5-3b", "phishing_email"): 0,   # benign
    ("llama3.2-3b", "sms_smishing"): 0,
    ("llama3.2-3b", "phishing_email"): 1,
    ("smollm2-1.7b", "sms_smishing"): 1,
    ("smollm2-1.7b", "phishing_email"): 0,
    ("phi3.5-mini", "sms_smishing"): 0,
    ("phi3.5-mini", "phishing_email"): 1,
}

# run_baseline.py sends the HF id (MODELS[key]); map it back to the short key
MODEL_ALIASES = {
    "qwen/qwen2.5-3b-instruct": "qwen2.5-3b",
    "qwen2.5-3b": "qwen2.5-3b",
    "meta-llama/llama-3.2-3b-instruct": "llama3.2-3b",
    "llama3.2-3b": "llama3.2-3b",
    "huggingfaceh4/smollm2-1.7b-instruct": "smollm2-1.7b",
    "smollm2-1.7b": "smollm2-1.7b",
    "microsoft/phi-3.5-mini-instruct": "phi3.5-mini",
    "phi3.5-mini": "phi3.5-mini",
}


def mock_key(model: str) -> str:
    m = (model or "").lower()
    return MODEL_ALIASES.get(m, m)


def resolve_model_id(model: str) -> str:
    """Accept either an HF id or a short study key ('qwen2.5-3b').

    Callers that forget the mapping used to get a 500 with "not a valid model
    identifier" from transformers, which looks like a broken model rather than a
    wrong name. Resolve it here instead.
    """
    name = (model or "").strip()
    if "/" in name and not name.lower().startswith(tuple(MODEL_ALIASES)):
        return name
    try:
        from src.common import MODELS
        if name in MODELS:
            return MODELS[name]
        lowered = {k.lower(): v for k, v in MODELS.items()}
        if name.lower() in lowered:
            return lowered[name.lower()]
    except Exception:
        pass
    return name


app = FastAPI(title="SLM Shim", version="0.3.0")


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: List[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8, ge=1, le=256)
    seed: int = Field(default=42, ge=0)
    task: str = Field(default="sms_smishing",
                      description="sms_smishing or phishing_email (mock lookup only)")


# model cache: one pipeline per model id, loaded lazily on first request
_PIPES = {}


def get_pipeline(model_id: str):
    """Load (once) and return a transformers text-generation pipeline.

    transformers 5 renamed `torch_dtype` to `dtype`. Passing the old kwarg is not
    ignored - it fails inside pipeline(), and the only thing the caller saw was
    an HTTP 500 from the shim (this is exactly what broke the first Kaggle run:
    Kaggle ships transformers 5.16.1). Choose the spelling by major version and
    fall back to the other one if the call still rejects it.
    """
    if model_id in _PIPES:
        return _PIPES[model_id]
    try:
        import torch
        import transformers
        from transformers import pipeline
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=501,
                            detail=f"Real mode needs torch+transformers: {e}")

    try:
        major = int(transformers.__version__.split(".")[0])
    except Exception:
        major = 4
    primary, fallback = ("dtype", "torch_dtype") if major >= 5 else ("torch_dtype", "dtype")
    # float16 has no CPU kernels for several ops: pick the dtype from the device
    # so the same shim runs on a T4 and on a CPU-only box (local smoke test).
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        _PIPES[model_id] = pipeline(
            "text-generation", model=model_id, device_map="auto",
            **{primary: dtype})
    except Exception as first:
        try:
            _PIPES[model_id] = pipeline(
                "text-generation", model=model_id, device_map="auto",
                **{fallback: dtype})
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=(f"pipeline() failed with {primary}={dtype}: {first}; "
                        f"retry with {fallback} also failed: "
                        f"{fallback}={dtype} rejected"))
    return _PIPES[model_id]


def _extract_text(gen):
    """Normalise a pipeline result to a plain string.

    transformers returns different shapes depending on version and input type:
      * str                      -> plain completion
      * {"role":..., "content":..}    -> single chat message
      * [msg, msg, ...]          -> chat message list (last one is the answer)
    """
    if isinstance(gen, str):
        return gen.strip()
    if isinstance(gen, dict):
        return (gen.get("content") or "").strip()
    if isinstance(gen, list):
        for item in reversed(gen):
            text = _extract_text(item)
            if text:
                return text
        return ""
    return str(gen).strip() if gen is not None else ""


@app.get("/health")
async def health():
    try:
        import torch  # noqa: F401
        torch_state = "yes" if torch.cuda.is_available() else "cpu-only"
    except ModuleNotFoundError:
        torch_state = "no"
    return JSONResponse(content={
        "status": "ok",
        "mode": "mock" if USE_MOCK else "real",
        "model": DEFAULT_MODEL,
        "torch": torch_state,
        "loaded_models": list(_PIPES),
    })


@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={"data": [
        {"id": DEFAULT_MODEL, "object": "model",
         "created": int(time.time()), "owned_by": "local"}
    ]})


@app.post("/v1/chat/completions")
async def chat_completion(req: ChatCompletionRequest):
    t0 = time.time()

    if USE_MOCK:
        # deterministic verdict from the research prompt + task
        user_msg = next((m.content for m in reversed(req.messages)
                         if m.role == "user"), "")
        msgs = build_messages(req.task, user_msg)
        prompt = msgs[1]["content"] if len(msgs) > 1 else user_msg
        verdict = MOCK_VERDICTS.get((mock_key(req.model), req.task), 0)
        raw_output = ("malicious" if verdict == 1 else "benign") + \
                     " (mock simulation, no model loaded)"
    else:
        # REAL mode: use the messages exactly as the caller built them, so the
        # chat template matches the research prompt (system + user).
        def _as_dict(m):
            # pydantic v2 has model_dump, v1 only has dict()
            fn = getattr(m, "model_dump", None) or m.dict
            return fn()

        chat = [_as_dict(m) for m in req.messages]
        gen_kwargs = {
            "max_new_tokens": req.max_tokens,
            "do_sample": req.temperature > 0,
            "return_full_text": False,
        }
        # temperature is only meaningful (and only accepted) when sampling
        if req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature

        try:
            pipe = get_pipeline(resolve_model_id(req.model))
            out = pipe(chat, **gen_kwargs)
            raw_output = _extract_text(out[0].get("generated_text"))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {type(e).__name__}: {e}")

    verdict_int = parse_verdict(raw_output)
    elapsed = round(time.time() - t0, 3)
    return JSONResponse(content={
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": raw_output},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0,
                  "completion_tokens": len(raw_output.split()),
                  "total_tokens": len(raw_output.split())},
        "shim": {"mode": "mock" if USE_MOCK else "real",
                 "parsed_verdict": verdict_int,
                 "elapsed_s": elapsed},
    })


# ────────────────────── verdict probabilities ──────────────────────
# The saliency attack (and the confidence-threshold defence) need P(malicious),
# not just the argmax word. This endpoint answers that question with ONE forward
# pass per text: it scores the first token of each allowed answer at the last
# prompt position, so no generation happens and it is several times cheaper
# than a chat completion.

class ProbRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    texts: List[str]
    task: str = Field(default="sms_smishing")
    batch_size: int = Field(default=8, ge=1, le=32)
    task_descriptions_ok: bool = Field(default=True,
        description="unused; kept so an older client payload still validates")


def _answer_token_ids(tok):
    """The one token per answer that carries the decision.

    The leading space matters: the model continues the prompt, so the answer
    starts with the space-prefixed form. But not every tokenizer spells that
    form the same way, and this bit us once in a way worth recording. Phi-3.5
    has no token for " malicious": it encodes it as [SPACE, "mal", "icious"]
    and " benign" as [SPACE, "ben", "ign"], so a naive `enc[0]` returns the
    SAME space token for both answers. The two logits are then identical, the
    softmax over the pair is exactly 0.5 for every text, and the whole
    probability path silently becomes a constant -- no error, no warning, and
    845/845 defence rows at p = 0.5.

    So: prefer the shortest encoding of each word, then skip the ids the two
    encodings share, and take the first position where they actually differ.
    If they never differ, fail loudly instead of returning 0.5 forever.
    """
    encs = {}
    for name in ("malicious", "benign"):
        variants = [list(e) for e in
                    (tok.encode(v, add_special_tokens=False)
                     for v in (" " + name, name)) if e]
        if not variants:
            raise HTTPException(status_code=501,
                                detail=f"tokenizer cannot encode {name!r}")
        encs[name] = min(variants, key=len)

    mal, ben = encs["malicious"], encs["benign"]
    i = 0
    while i < min(len(mal), len(ben)) and mal[i] == ben[i]:
        i += 1
    if i >= min(len(mal), len(ben)):
        raise HTTPException(
            status_code=501,
            detail=("the answers share a full token prefix (%r vs %r): a "
                    "single-token verdict probability is not defined for "
                    "this tokenizer" % (mal, ben)))
    ids = {"malicious": int(mal[i]), "benign": int(ben[i]),
           "offset": i}
    return ids


def _accepted_kwargs(model):
    """Names the model's forward() accepts, so encoder metadata is not passed in.

    Some tokenizers return token_type_ids (Phi-3 does) and several architectures
    reject them with a TypeError. Filtering here is cheaper than a try/except
    around the call, and it fails visibly if even input_ids is missing.
    """
    import inspect

    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return None                      # unknown: pass everything through
    names = set(sig.parameters)
    if not ("input_ids" in names or "inputs_embeds" in names):
        return None
    names.add("kwargs")
    return names


def _probs_real(model_id, texts, task, batch_size):
    import torch

    pipe = get_pipeline(model_id)
    tok, model = pipe.tokenizer, pipe.model
    ids = _answer_token_ids(tok)
    # left padding puts the real last token at index -1 for every row, which is
    # where the answer logits live
    prev_side = tok.padding_side
    prev_trunc = getattr(tok, "truncation_side", "right")
    tok.padding_side = "left"
    # truncation, when it happens, must drop the FRONT of a long message: the
    # instruction that asks for a one-word verdict is at the END of the prompt,
    # and cutting it would score a different question
    tok.truncation_side = "left"
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    accepted = _accepted_kwargs(model)
    out = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            prompts = [tok.apply_chat_template(
                build_messages(task, t), tokenize=False,
                add_generation_prompt=True) for t in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048)
            if accepted is not None:
                enc = {k: v for k, v in enc.items()
                       if k in accepted or k in ("input_ids", "attention_mask")}
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
            mal = logits[:, ids["malicious"]]
            ben = logits[:, ids["benign"]]
            mal_list, ben_list = mal.tolist(), ben.tolist()
            # The v4 failure mode: identical logits would make P(malicious) a
            # constant 0.5. That is not a measurement, so it is an error.
            if len(chunk) > 1 and all(a == b for a, b in
                                      zip(mal_list, ben_list)):
                raise HTTPException(
                    status_code=500,
                    detail=("the two answer logits are identical for every "
                            "row: the shim is not scoring a decision"))
            # normalise over the two allowed answers only: this is P(malicious)
            # under the task the prompt defines, not a full-vocabulary softmax
            p_mal = torch.softmax(torch.stack([mal, ben], dim=1), dim=1)[:, 0]
            for text, p, a, b in zip(chunk, p_mal.tolist(),
                                     mal_list, ben_list):
                out.append({"text": text, "p_malicious": round(float(p), 6),
                            "logit_malicious": round(float(a), 4),
                            "logit_benign": round(float(b), 4),
                            "p_benign": round(1.0 - float(p), 6)})
    finally:
        tok.padding_side = prev_side
        tok.truncation_side = prev_trunc
    return out


# Every codepoint the char-level attack can insert. The mock shim lowers its
# confidence when it sees one, so a local smoke test exercises the same code
# path a real run does instead of returning a constant for every text.
_MOCK_ATTACK_MARKS = "\u0430\u0435\u043e\u0440\u0441\u0445\u0443\u0456\u0455\u0410\u0412\u0415"


def _probs_mock(model, task, texts):
    key = mock_key(model)
    verdict = MOCK_VERDICTS.get((key, task), 0)
    out = []
    for text in texts:
        h = (abs(hash((key, task, text))) % 1000) / 1000.0
        p = (0.70 + 0.29 * h) if verdict == 1 else (0.01 + 0.20 * h)
        if any(ch in text for ch in _MOCK_ATTACK_MARKS):
            p = min(p, 0.25)
        out.append({"text": text, "p_malicious": round(p, 6),
                    "p_benign": round(1.0 - p, 6),
                    "logit_malicious": None, "logit_benign": None})
    return out


@app.post("/v1/verdict_probs")
async def verdict_probs(req: ProbRequest):
    t0 = time.time()
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    try:
        if USE_MOCK:
            rows = _probs_mock(req.model, req.task, req.texts)
        else:
            rows = _probs_real(resolve_model_id(req.model), req.texts,
                               req.task, req.batch_size)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"verdict_probs failed: {type(e).__name__}: {e}")
    return JSONResponse(content={
        "model": req.model,
        "task": req.task,
        "probabilities": rows,
        "shim": {"mode": "mock" if USE_MOCK else "real",
                 "elapsed_s": round(time.time() - t0, 3)},
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    mode = "mock" if USE_MOCK else "real"
    print(f"[shim] starting on http://localhost:{port} (mode={mode})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
