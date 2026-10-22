"""
Open paper/main.docx in the installed Word and export a PDF.

    python src/render_docx_check.py

This is a compatibility check, not a build step: python-docx writes a package
that no Word process has ever opened, and "the XML is well formed" is not the
same claim as "Word opens it and lays it out". The PDF it writes
(paper/main.word-preview.pdf) is only for looking at; the typeset artefact is
still paper/main.pdf from LaTeX.

If Word is not installed or cannot be automated, this exits 2 and says so
instead of pretending the file was checked.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX = os.path.join(ROOT, "paper", "main.docx")
PDF = os.path.join(ROOT, "paper", "main.word-preview.pdf")


def main():
    if not os.path.exists(DOCX):
        print("missing %s - run src/make_docx.py first" % DOCX)
        return 1
    try:
        import win32com.client
    except ImportError:
        print("SKIPPED: pywin32 not installed, cannot automate Word")
        return 2
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(DOCX, ReadOnly=True)
        pages = doc.ComputeStatistics(2)          # wdStatisticPages
        words = doc.ComputeStatistics(0)          # wdStatisticWords
        doc.ExportAsFixedFormat(PDF, 17)          # wdExportFormatPDF
        doc.Close(False)
        print("Word opened %s: %d pages, %d words" % (os.path.basename(DOCX),
                                                      pages, words))
        print("wrote %s" % PDF)
        return 0
    except Exception as exc:                       # noqa: BLE001
        print("FAILED to open the .docx in Word: %s" % exc)
        return 1
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:                      # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(main())
