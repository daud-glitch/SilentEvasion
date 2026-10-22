"""Split sections/09_transfer.tex into 09/10/11 and guard the generated inputs.

One-off helper kept in the repo tree so the split is reproducible rather than a
hand edit of three files: the section text is written once, and every
\\input of a machine-generated block is wrapped in \\IfFileExists so a PDF can
never include a table that no run produced.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SECTIONS = os.path.join(ROOT, "paper", "sections")
SEP = "\n" + chr(92) + "section{"

GUARDED = {
    "sections/gen_transfer_results": "tables/t5_transfer.tex",
    "sections/gen_saliency_results": "tables/t7_saliency.tex",
    "sections/gen_defense_results": "tables/t6_defense.tex",
}
TABLES = ["tables/t5_transfer", "tables/t7_saliency", "tables/t6_defense"]


def guard(line):
    for key, table in GUARDED.items():
        if line.strip() == (chr(92) + "input{" + key + "}"):
            return (chr(92) + "IfFileExists{" + table +
                    "}{" + chr(92) + "input{" + key + "}}{}")
    return line


def main():
    src = os.path.join(SECTIONS, "09_transfer.tex")
    text = open(src, encoding="utf-8").read()
    parts = text.split(SEP)
    if len(parts) != 3:
        raise SystemExit("expected 3 sections, found %d" % len(parts))
    names = ["09_transfer.tex", "10_saliency.tex", "11_defense.tex"]
    for i, (name, body) in enumerate(zip(names, parts)):
        # str.split() consumed the separator, which carried the heading of every
        # section except the first - put it back or 09 and 10 lose their titles
        if i:
            body = SEP + body
        body = body.lstrip("\n")
        lines = [guard(ln) for ln in body.split("\n")]
        out = "\n".join(lines).rstrip() + "\n"
        with open(os.path.join(SECTIONS, name), "w", encoding="utf-8",
                  newline="\n") as f:
            f.write(out)
        print("wrote %s (%d words)" % (name, len(out.split())))


if __name__ == "__main__":
    main()
