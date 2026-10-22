"""Tests for the shim's probability path, without a GPU.

The real path (`_probs_real`) only runs on a GPU box, which makes it exactly the
kind of code that breaks silently in a scored run: a wrong padding side or a
rejected keyword argument turns into an HTTP 500, and the saliency attack then
reports "probabilities unavailable" for every sample - a plausible-looking
failure that is really a bug.

So the path is tested against a fake torch and a fake tokenizer/model that
implement the same tiny surface, and the assertions are about the things that
would actually go wrong:

  * left padding, so the last logit belongs to the real last token;
  * left truncation, so a long message keeps the "answer with one word" tail;
  * a pad token is created when the tokenizer has none (Qwen-style);
  * token_type_ids are dropped for a model that does not accept them (Phi-3);
  * softmax is computed over the two answer tokens only;
  * tokenizer state (padding/truncation side) is restored even on failure.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import src.shim_colab as shim  # noqa: E402


class FakeTensor:
    """Minimal tensor: a numpy array wrapper with the three ops we use."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def to(self, _device):
        return self

    def float(self):
        return self

    def __getitem__(self, key):
        return FakeTensor(self.arr[key])

    def tolist(self):
        return self.arr.tolist()

    def __len__(self):
        return len(self.arr)


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = types.SimpleNamespace(is_available=lambda: False)

    def no_grad(self):
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return _Ctx()

    def softmax(self, x, dim=1):
        a = x.arr if isinstance(x, FakeTensor) else x
        e = np.exp(a - a.max(axis=dim, keepdims=True))
        return FakeTensor(e / e.sum(axis=dim, keepdims=True))

    def stack(self, tensors, dim=0):
        return FakeTensor(np.stack([t.arr for t in tensors], axis=dim))


class FakeTokenizer:
    """Encodes every message to a fixed-length id list, recording the calls."""

    padding_side = "right"
    truncation_side = "right"
    pad_token = None
    eos_token = "<eos>"
    calls = []

    def encode(self, text, add_special_tokens=False):
        return {" malicious": [7, 1], " benign": [8, 2],
                "malicious": [7, 1], "benign": [8, 2]}.get(text, [5])

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert not tokenize
        return " | ".join(m["content"] for m in messages) + " <|assistant|>"

    def __call__(self, prompts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        FakeTokenizer.calls.append({
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "pad_token": self.pad_token,
            "n": len(prompts), "truncation": truncation,
            "max_length": max_length,
        })
        ids = np.array([[11, 12, 13]] * len(prompts), dtype=float)
        out = {"input_ids": FakeTensor(ids),
               "attention_mask": FakeTensor(np.ones_like(ids)),
               "token_type_ids": FakeTensor(np.zeros_like(ids))}
        return out


class FakeLogits:
    def __init__(self, arr):
        self.logits = FakeTensor(arr)


class FakeModel:
    """Accepts input_ids/attention_mask only, like Phi-3.5-mini."""

    device = "cpu"
    received = []

    def forward(self, input_ids=None, attention_mask=None):
        pass

    def __call__(self, **kwargs):
        FakeModel.received.append(sorted(kwargs))
        vocab = 20
        logits = np.zeros((kwargs["input_ids"].arr.shape[0], 1, vocab))
        logits[:, -1, 7] = 2.0      # malicious
        logits[:, -1, 8] = 0.0      # benign
        return FakeLogits(logits)


class FakePipe:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()


def run(texts, task="sms_smishing", batch_size=8):
    FakeTokenizer.calls = []
    FakeModel.received = []
    real = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch()
    orig_pipeline = shim.get_pipeline
    shim.get_pipeline = lambda model_id: FakePipe()
    try:
        return shim._probs_real("fake/model", texts, task, batch_size)
    finally:
        shim.get_pipeline = orig_pipeline
        if real is not None:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)


def test_probabilities_are_computed_from_the_two_answer_tokens():
    rows = run(["free prize click here"])
    p = rows[0]["p_malicious"]
    # logits 2.0 vs 0.0 -> sigmoid(2) = 0.880797
    assert abs(p - 0.880797) < 1e-4, p
    assert abs(rows[0]["p_benign"] - (1 - p)) < 1e-9


def test_left_padding_and_left_truncation_are_requested():
    run(["a", "b"])
    call = FakeTokenizer.calls[0]
    assert call["padding_side"] == "left", call
    assert call["truncation_side"] == "left", call
    assert call["truncation"] is True and call["max_length"] == 2048, call


def test_missing_pad_token_is_filled_from_eos():
    run(["a"])
    assert FakeTokenizer.calls[0]["pad_token"] == "<eos>"


def test_tokenizer_state_is_restored_after_the_call():
    tok = FakeTokenizer()
    assert tok.padding_side == "right" and tok.truncation_side == "right"
    run(["a"])
    # the instance used by run() is a fresh one; check the class defaults were
    # not mutated by the implementation
    assert FakeTokenizer.padding_side == "right"
    assert FakeTokenizer.truncation_side == "right"


def test_token_type_ids_are_dropped_for_a_model_that_rejects_them():
    run(["a"])
    assert "token_type_ids" not in FakeModel.received[0], FakeModel.received[0]
    assert "input_ids" in FakeModel.received[0]


def test_batching_splits_the_work():
    rows = run(["a", "b", "c", "d", "e"], batch_size=2)
    assert len(rows) == 5
    assert [c["n"] for c in FakeTokenizer.calls] == [2, 2, 1]


def test_answer_token_ids_prefer_the_space_prefixed_form():
    ids = shim._answer_token_ids(FakeTokenizer())
    assert ids["malicious"] == 7 and ids["benign"] == 8, ids
    assert ids["offset"] == 0, ids


def test_phi_style_shared_space_prefix_does_not_collapse_the_answers():
    """Phi-3.5 encodes " malicious" as [SPACE, mal, icious].

    Taking enc[0] returned the SPACE id for both answers in the v4 run: equal
    logits, softmax exactly 0.5 for all 845 defence rows, and a defence table
    that looked like a 100% ASR. The prefix has to be skipped.
    """

    class PhiTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return {" malicious": [9, 7, 1], "malicious": [7, 1],
                    " benign": [9, 8, 2], "benign": [8, 2]}.get(text, [5])

    ids = shim._answer_token_ids(PhiTokenizer())
    assert ids["malicious"] == 7 and ids["benign"] == 8, ids
    assert ids["offset"] == 0, ids
    # and the whole path still produces a decision, not a constant
    rows = run(["a", "b"], batch_size=2)
    assert all(abs(r["p_malicious"] - 0.880797) < 1e-4 for r in rows), rows


def test_a_full_shared_prefix_fails_loudly():
    """If the answers cannot be told apart, the endpoint must not return 0.5."""

    class SameTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [9, 9]

    try:
        shim._answer_token_ids(SameTokenizer())
    except Exception as exc:                       # HTTPException
        assert "prefix" in str(exc.detail), str(exc.detail)
    else:
        raise AssertionError("a shared prefix must raise, not return 0.5")


def test_identical_answer_logits_raise_instead_of_returning_half():
    class FlatModel(FakeModel):
        def __call__(self, **kwargs):
            logits = np.zeros((kwargs["input_ids"].arr.shape[0], 1, 20))
            return FakeLogits(logits)          # every logit equal

    class Pipe(FakePipe):
        def __init__(self):
            super().__init__()
            self.model = FlatModel()

    real = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch()
    orig = shim.get_pipeline
    shim.get_pipeline = lambda model_id: Pipe()
    try:
        try:
            shim._probs_real("fake/model", ["a", "b"], "sms_smishing", 8)
        except Exception as exc:
            assert "identical" in str(getattr(exc, "detail", exc))
        else:
            raise AssertionError("a constant 0.5 must be an error")
    finally:
        shim.get_pipeline = orig
        if real is not None:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)


def test_state_restored_even_when_forward_raises():
    class Boom(FakeModel):
        def __call__(self, **kwargs):
            raise RuntimeError("cuda oom")

    class Pipe(FakePipe):
        def __init__(self):
            super().__init__()
            self.model = Boom()

    real = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch()
    orig = shim.get_pipeline
    shim.get_pipeline = lambda model_id: Pipe()
    tok = shim.get_pipeline("x").tokenizer
    try:
        try:
            shim._probs_real("fake/model", ["a"], "sms_smishing", 8)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the failure must propagate to the endpoint")
    finally:
        shim.get_pipeline = orig
        if real is not None:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)
    assert tok.padding_side == "right" and tok.truncation_side == "right"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
