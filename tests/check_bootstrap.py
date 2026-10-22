"""
Guard test: colab_bootstrap.py must match the current src/ + tests/ sources.

Why this exists: the Colab cell is a *copy* of the sources. If someone fixes a
bug in src/ and forgets `python src/make_bootstrap.py`, Colab silently keeps
running the old code - which is how a fixed shim looked broken for three runs
while the notebook simply had no such file.

Checks:
  1. the checked-in cell is byte-identical to a freshly rendered one
  2. its embedded payload decompresses and every file matches its sha256
  3. the manifest covers every current src/*.py and tests/*.py
  4. its expected split fingerprints match what this machine builds

Run:  python tests/check_bootstrap.py
"""
import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import make_bootstrap as mb  # noqa: E402


def load_generated():
    """Import the cell as a module. main() is guarded by __name__, so it won't run."""
    spec = importlib.util.spec_from_file_location("slm_bootstrap_cell", mb.OUT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    problems = []
    notes = []

    if not os.path.exists(mb.OUT):
        print("MISSING colab_bootstrap.py -> run: python src/make_bootstrap.py")
        return 1

    files = mb.collect_files()
    with open(mb.OUT, "r", encoding="utf-8") as f:
        on_disk = f.read()
    fresh = mb.render(files, mb.build_payload(files), mb.split_fingerprints())
    if fresh != on_disk:
        problems.append("colab_bootstrap.py is STALE vs src/ and tests/ "
                        "-> run: python src/make_bootstrap.py")
    else:
        notes.append(f"cell up to date ({len(on_disk)} bytes, {len(files)} files)")

    mod = load_generated()

    # 3. manifest covers exactly the current sources
    expected = sorted(files)
    got = sorted(mod.FILES_SHA256)
    if got != expected:
        only_disk = sorted(set(expected) - set(got))
        only_cell = sorted(set(got) - set(expected))
        problems.append(f"manifest mismatch: missing from cell={only_disk} "
                        f"extra in cell={only_cell}")
    else:
        notes.append(f"manifest covers all {len(expected)} source files")

    # 2. payload integrity
    import base64
    import gzip
    import io
    import tarfile
    try:
        blob = gzip.decompress(base64.b64decode(mod.PAYLOAD_B64))
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            members = {m.name: tar.extractfile(m).read()
                       for m in tar.getmembers() if m.isfile()}
    except Exception as exc:  # corrupt payload must fail loudly, not pass
        problems.append(f"payload unreadable: {type(exc).__name__}: {exc}")
        members = {}

    if members:
        payload_ok = True
        for rel, data in sorted(members.items()):
            want = mod.FILES_SHA256.get(rel)
            got_hash = hashlib.sha256(data).hexdigest()
            with open(os.path.join(ROOT, rel), "rb") as f:
                disk = f.read()
            if want != got_hash:
                problems.append(f"{rel}: payload sha256 != manifest")
                payload_ok = False
            elif data != disk:
                problems.append(f"{rel}: payload differs from working tree")
                payload_ok = False
        if payload_ok:
            notes.append(f"payload verified ({len(members)} files, "
                         f"{sum(len(d) for d in members.values())} bytes)")

    # 4. split fingerprints
    current = mb.split_fingerprints()
    if mod.EXPECTED_SPLITS != current:
        problems.append("expected split fingerprints differ from what this "
                        "machine builds -> the frozen splits changed")
    else:
        notes.append("split fingerprints match the embedded expectations")

    # 5. the runbook notebook / checklist must carry this same cell
    nb_path = os.path.join(ROOT, "colab_runbook.ipynb")
    md_path = os.path.join(ROOT, "COLAB_STEPS.md")
    for path in (nb_path, md_path):
        if not os.path.exists(path):
            problems.append(f"missing {os.path.basename(path)} "
                            f"-> run: python src/make_runbook.py")
    if os.path.exists(nb_path):
        import make_runbook as mr
        with open(nb_path, encoding="utf-8") as f:
            on_disk_nb = f.read()
        fresh_nb = json.dumps(mr.build_notebook(), indent=1, ensure_ascii=False) + "\n"
        if on_disk_nb != fresh_nb:
            problems.append("colab_runbook.ipynb is STALE "
                            "-> run: python src/make_runbook.py")
        else:
            notes.append("runbook notebook reflects the current sources")

    # 6. the Kaggle kernel must embed the same bootstrap
    kernel_path = os.path.join(ROOT, "kaggle", "kernel.py")
    if not os.path.exists(kernel_path):
        problems.append("missing kaggle/kernel.py -> run: python src/make_kaggle.py")
    else:
        import make_kaggle as mk
        with open(kernel_path, encoding="utf-8") as f:
            kernel_text = f.read()
        if mk.embed_bootstrap() not in kernel_text:
            problems.append("kaggle/kernel.py is STALE "
                            "-> run: python src/make_kaggle.py")
        else:
            notes.append("kaggle kernel embeds the current bootstrap")

    print("BOOTSTRAP-CELL CHECK")
    for n in notes:
        print(f"  ok   {n}")
    for p in problems:
        print(f"  FAIL {p}")
    print("RESULT:", "OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
