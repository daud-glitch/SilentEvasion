"""Ad-hoc check of the shim's real-inference branch without torch/GPU.

Stubs get_pipeline, then asserts the caller's messages reach the pipeline
unmodified (no double-wrapped prompt, system role preserved) and that the
response is normalised + parsed correctly.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.shim_colab as S

assert S.USE_MOCK is False, "run with SHIM_MODE=real"


class FakePipe:
    def __call__(self, chat, **kw):
        assert chat[0]["role"] == "system", chat[0]
        user = chat[1]["content"]
        assert user.count("BEGIN MESSAGE") == 1, f"prompt was double-wrapped: {user!r}"
        assert kw["do_sample"] is False
        assert "temperature" not in kw, "temperature must be omitted when not sampling"
        return [{"generated_text": {"role": "assistant", "content": "  benign \n"}}]


S.get_pipeline = lambda mid: FakePipe()

# exactly what run_baseline.py sends: the full research prompt as chat messages
msgs = S.build_messages("sms_smishing", "Free prize winner claim now")
req = S.ChatCompletionRequest(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=msgs,
    temperature=0.0,
    max_tokens=8,
    task="sms_smishing",
)
body = json.loads(asyncio.run(S.chat_completion(req)).body)

print("content   :", repr(body["choices"][0]["message"]["content"]))
print("shim block:", body["shim"])
assert body["choices"][0]["message"]["content"] == "benign"
assert body["shim"]["parsed_verdict"] == 0
assert body["shim"]["mode"] == "real"
print("REAL-BRANCH OK")
