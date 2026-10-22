# Bibliography verification

Every entry in `refs.bib` was opened on a primary source before it was added.
Nothing here is reconstructed from memory. The column on the right is the page
that was actually read.

| key | read at | what the page verified |
|---|---|---|
| `gao2018deepwordbug` | https://arxiv.org/abs/1801.04354 | authors, title, arXiv id; page states it is an extended version of a workshop paper (the workshop's own proceedings record was **not** readable, so the entry is the arXiv version) |
| `jin2020textfooler` | https://ojs.aaai.org/index.php/AAAI/article/view/6311 | authors, title, AAAI vol. 34 no. 05, pages 8018--8025, DOI |
| `ren2019pwws` | https://aclanthology.org/P19-1103/ | authors, title, ACL 2019, pages 1085--1097, DOI |
| `morris2020textattack` | https://aclanthology.org/2020.emnlp-demos.16/ | authors, title, EMNLP demos, pages 119--126, DOI |
| `ebrahimi2018hotflip` | https://aclanthology.org/P18-2006/ | authors, title, ACL 2018 short papers, pages 31--36, DOI |
| `rojas2024zeroshot` | https://arxiv.org/abs/2405.15936 | authors, title, WEA 2024 DOI, reported F1 figures |
| `eilertsen2025intent` | https://arxiv.org/html/2506.14337v1 | authors, title, zero/few-shot phishing prompt results |
| `koide2024chatspamdetector` | https://arxiv.org/abs/2402.18093 | authors, title, SecureComm 2024, 99.70% accuracy claim |
| `wang2025smishx` | https://www.usenix.org/conference/soups2025/presentation/wang | authors, SOUPS 2025, 98.8% accuracy claim |
| `thapa2025quantized` | https://arxiv.org/html/2507.07406v1 | authors, title, quantised-LLM comparison, rephrasing degradation |
| `shafee2026falsealarms` | https://arxiv.org/html/2507.06252v2 | authors, journal DOI, 75% FPR claim for ChatGPT-4o |
| `kulkarni2024mltollm` | https://arxiv.org/abs/2407.20361 | authors, title, webpage-adversarial findings |
| `carlini2019evaluating` | https://arxiv.org/abs/1902.06705 | authors, title, scope of the methodology paper |
| `zimmermann2022confidence` | https://proceedings.neurips.cc/paper_files/paper/2022/hash/5545d9bcefb7d03d5ad39a905d14fbe3-Abstract-Conference.html | authors, NeurIPS 2022, "11 of 13" claim |
| `yuan2021transfer` | https://aclanthology.org/2021.emnlp-main.121/ | authors, title, EMNLP 2021, DOI |
| `he2021extraction` | https://aclanthology.org/2021.naacl-main.161/ | authors, title, NAACL 2021, DOI |
| `voracek2024statistical` | https://arxiv.org/abs/2406.17830 | authors, NeurIPS 2024, binomial-interval scope |
| `nistwilson` | https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm | Wilson method for proportion intervals, boundary behaviour |
| `miller1995wordnet` | https://api.crossref.org/works/10.1145/219717.219748 | DOI registry record: ACM journal article, *Communications of the ACM* vol. 38 issue 11, pages 39--41, published November 1995, title and DOI |

## Known gaps, stated rather than papered over

- **DeepWordBug**: only the arXiv record is verified; the workshop proceedings
  citation was not readable, so the paper cites the arXiv version.
- **Kulkarni et al.**: the ACM DOI page returned 403; only the arXiv record is
  cited.
- **SmishX**: the SOUPS PDF itself was not readable; the USENIX presentation
  page was, and the 98.8% figure comes from that page's summary.
- **WordNet (Miller)**: the ACM Digital Library page returned 403, so the entry
  is verified against the DOI's Crossref record rather than the article page.
  Volume, issue, pages, publisher and date all come from that record; the
  abstract text is Crossref's copy of the published abstract.
- **No paper was found** that prescribes Wilson intervals specifically for
  conditional attack success rates. The text therefore cites the NIST handbook
  for the statistical method and the adversarial-robustness methodology papers
  for the evaluation discipline, and does not attribute the interval choice to
  prior security work.
