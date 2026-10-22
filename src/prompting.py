"""
Unified zero-shot prompt + verdict parser for SLM security triage.

DESIGN RULES (do not change casually - consistency is what makes results comparable):
  * One prompt template for every model. No model-specific tweaking, no few-shot.
    If a model does badly, that is a RESULT, not a bug to fix.
  * temperature = 0, do_sample = False, max_new_tokens = 8.
    We need a deterministic single-word verdict, nothing else.
  * The prompt never reveals dataset-specific hints ("this is an SMS", "this is an email").
    It only describes the security decision, so the same template works across tasks.
"""

TASK_DESCRIPTIONS = {
    "sms_smishing": "an SMS text message that arrived on a mobile phone",
    "phishing_email": "an email message that arrived in a corporate inbox",
}

SYSTEM_PROMPT = (
    "You are a security triage classifier used by a Security Operations Centre. "
    "Your job is to decide whether a single message is malicious (phishing, smishing, "
    "scam, credential theft, or malware delivery) or benign. "
    "Reply with exactly one word: malicious or benign. Do not explain."
)


def build_messages(task: str, text: str) -> list[dict]:
    """Build the chat message list for one sample."""
    kind = TASK_DESCRIPTIONS.get(task, "a message")
    user = (
        f"The following is {kind}. Decide if it is malicious or benign.\n\n"
        "----- BEGIN MESSAGE -----\n"
        f"{text}\n"
        "----- END MESSAGE -----\n\n"
        "Answer with exactly one word: malicious or benign."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_verdict(raw: str) -> int | None:
    """
    Map model text output to a binary label.
      1 = malicious, 0 = benign, None = unparseable (counted separately, never guessed).
    """
    if not raw:
        return None
    s = raw.strip().lower()
    # Look at the first 40 chars only: any later mention of the other class is not the verdict.
    head = s[:40]
    if "malicious" in head:
        return 1
    if "benign" in head:
        return 0
    # Fallback for terse models that answer "yes"/"no" or "spam"/"ham"
    if head.startswith("spam") or head.startswith("phish"):
        return 1
    if head.startswith("ham") or head.startswith("legit") or head.startswith("safe"):
        return 0
    return None
