"""Render the TikZ pipeline figure to the PNG that the Word build embeds.

    python src/make_pipeline_png.py

The typeset paper draws figures/fig_pipeline_body.tex directly, so it is always
current. Word cannot read TikZ, so the same file is compiled here in a
standalone document and rasterised to figures/fig_pipeline.png at 300 dpi,
then trimmed to its ink. One source, two builds.

Needs the bundled tectonic (for the PDF) and pypdfium2 + Pillow (to rasterise).
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
BODY = os.path.join(FIG, "fig_pipeline_body.tex")
PNG = os.path.join(FIG, "fig_pipeline.png")
TECTONIC = os.path.join(ROOT, ".tools", "tectonic.exe")

WRAPPER = r"""\documentclass[border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\begin{document}
\input{fig_pipeline_body}
\end{document}
"""


def build_pdf(tmp):
    """Compile the shared TikZ body on its own and return the PDF path."""
    shutil.copyfile(BODY, os.path.join(tmp, "fig_pipeline_body.tex"))
    tex = os.path.join(tmp, "pipeline.tex")
    with open(tex, "w", encoding="utf-8") as fh:
        fh.write(WRAPPER)
    proc = subprocess.run([TECTONIC, "-X", "compile", "pipeline.tex"],
                          cwd=tmp, capture_output=True, text=True)
    pdf = os.path.join(tmp, "pipeline.pdf")
    if proc.returncode != 0 or not os.path.exists(pdf):
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("tectonic failed to build the pipeline figure")
    return pdf


def main():
    if not os.path.exists(BODY):
        print("missing %s" % BODY)
        return 1
    if not os.path.exists(TECTONIC):
        print("missing %s" % TECTONIC)
        return 1
    try:
        import pypdfium2 as pdfium
        from PIL import Image
    except ImportError:
        print("SKIPPED: need pypdfium2 and Pillow to rasterise the figure "
              "(python -m pip install pypdfium2 pillow)")
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        pdf = build_pdf(tmp)
        doc = pdfium.PdfDocument(pdf)
        try:
            image = doc[0].render(scale=300 / 72).to_pil().convert("RGB")
        finally:
            doc.close()          # Windows: the handle must go before cleanup
        ink = image.convert("L").point(lambda v: 255 - v).getbbox()
        if ink:
            pad = 12
            image = image.crop((max(0, ink[0] - pad), max(0, ink[1] - pad),
                                min(image.width, ink[2] + pad),
                                min(image.height, ink[3] + pad)))
        image.save(PNG, dpi=(300, 300))

    with Image.open(PNG) as im:
        print("wrote %s (%d x %d px, %.1f x %.1f cm at 300 dpi)"
              % (os.path.relpath(PNG, ROOT), im.width, im.height,
                 im.width / 300 * 2.54, im.height / 300 * 2.54))
    return 0


if __name__ == "__main__":
    sys.exit(main())
