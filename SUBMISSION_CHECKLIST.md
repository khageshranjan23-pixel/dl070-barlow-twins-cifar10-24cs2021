# DL-070 Deep Learning Laboratory: Submission Checklist

**Student Name:** Khagesh Ranjan  
**Roll Number:** 24cs2021  
**Institution:** Rajiv Gandhi Institute of Petroleum Technology (RGIPT)  
**Department:** Department of Computer Science and Engineering  
**Course:** DL-070 Deep Learning Laboratory  
**Topic:** Reduced-Scale Reproduction and Ablation Study of Barlow Twins on CIFAR-10  
**Submission Package:** `DL070_Barlow_Twins_Submission/` and `DL070_Barlow_Twins_Submission.zip`  
**Date:** October 2026  
**Status:** **100% COMPLETE & VERIFIED**

---

## Deliverable Verification Matrix

| Category | Deliverable Item | Location / Path | Status | Verification Details |
| :--- | :--- | :--- | :--- | :--- |
| **Code** | Modular Engine | `src/barlow_local.py` | Verified | Clean, readable PyTorch code; pretrain/eval CLI; ResNet-18 CIFAR adaptations; AMP support; 13 structured sections. |
| **Tests** | Unit Test Suite | `tests/test_core.py` | Verified | 8 comprehensive tests (standardization, off-diag, cross-corr, loss, no_rr, view swap, autograd, checkpoint). 100% pass rate. |
| **Notebook** | Google Colab | `notebooks/DL070_BarlowTwins_Colab.ipynb` | Verified | Structured and statically validated for execution in a clean Colab GPU session; zero local Windows paths; runnable on free cloud T4 GPU. |
| **Report** | PDF Conference Paper | `report/DL070_BarlowTwins_Report.pdf` | Verified | Exactly 4 pages; IEEE two-column format; clean column balancing; figures & tables embedded; student attribution. |
| **Report** | LaTeX Source | `report/DL070_BarlowTwins_Report.tex` | Verified | Compilable IEEEtran source code. |
| **Report** | BibTeX Citations | `report/references.bib` | Verified | 10 formal bibliographic citations matching in-text references. |
| **Presentation**| Slide Deck | `presentation/DL070_BarlowTwins_Presentation.pptx`| Verified | 12 professional 16:9 widescreen slides; navy/slate theme; high-contrast tables; student attribution. |
| **Presentation**| Speaker Script | `presentation/PRESENTATION_SCRIPT.md` | Verified | 10-minute slide transcript with timing markers, speaker notes, and slide transitions. |
| **Results** | Metrics JSON | `results/results_summary.json` | Verified | Primary numbers verified against raw disk logs (`bt_s0`: 74.84% (120.2 min, 537 MB), `no_rr`: 57.12% (93.1 min, 602 MB), `fixed`: 10.00% (76.4 min, 537 MB), `scalar`: 71.15% (87.0 min, 537 MB), `noise`: 72.69% (88.3 min, 537 MB)). |
| **Results** | Summary Report | `results/RESULTS_SUMMARY.md` | Verified | Complete markdown results document with full quantitative comparisons and limitations. |
| **Results** | Loss Dynamics Figure | `results/figures/loss_curves.png` | Verified | Convergence of total, invariance, and redundancy reduction loss over 50 epochs. |
| **Results** | Ablation Bar Chart | `results/figures/ablation_comparison.png` | Verified | Accuracy comparison and feature representation rank across all ablations and controls. |
| **Viva** | Code Walkthrough | `viva/CODE_WALKTHROUGH.md` | Verified | Comprehensive architecture walkthrough, module breakdowns, and code tracing. |
| **Viva** | Q&A Defense Guide | `viva/VIVA_QUESTIONS_AND_ANSWERS.md` | Verified | 15+ rigorous viva defense questions, answers, and 2 blackboard mathematical derivations. |
| **Meta** | Submission README | `README.md` | Verified | Full installation guide, quickstart, reproduction commands, and student candidate details. |
| **Meta** | Dependencies | `requirements.txt` | Verified | Clean, platform-agnostic dependencies (PyTorch, torchvision, reportlab, python-pptx, pymupdf). |
| **Meta** | License & Citations | `LICENSE_OR_CITATION_NOTES.md` | Verified | MIT License terms with Khagesh Ranjan copyright and academic attribution citations. |
| **Archive** | Clean ZIP Package | `DL070_Barlow_Twins_Submission.zip` | Verified | Compressed package; excludes `_archive_not_for_submission`, `.pyc`, and raw checkpoints. |

---

## Scientific Rigour and Audit Compliance

- [x] **Clean Process Termination:** Confirmed all exploratory background tasks cleanly terminated without data corruption or partial writes.
- [x] **Runtime Consistency Verified:** Pretraining times and peak VRAM directly verified from raw artifacts: `bt_s0` (120.2 min, 537 MB), `no_rr_s0` (93.1 min total / 21.0 min resumed, 602 MB), `fixed_s0` (76.4 min, 537 MB), `scalar_s0` (87.0 min, 537 MB), `noise_s0` (88.3 min, 537 MB) consistently applied across all deliverables.
- [x] **`no_rr` Interpretation Corrected:** Replaced "informational collapse" with approved exact phrasing: *"Removing redundancy reduction substantially reduced downstream discriminability despite preserving high numerical feature rank. This shows that full rank alone does not guarantee semantically useful or sufficiently decorrelated representations."*
- [x] **Dimensional Collapse Reserved:** "In this reduced-scale setup, removing redundancy reduction substantially reduced downstream discriminability. The tested static-threshold configuration produced rank-0 collapse, showing that this particular thresholding strategy failed to maintain a useful representation."
- [x] **Overclaims & Terminology Harmonized:** Replaced "Full reproduction" with "Reduced-scale CIFAR-10 reproduction", "publication-grade IEEE paper" with "IEEE-formatted assignment report", "full-batch standardization" with "minibatch feature standardization", and "Continuous Repulsion" with "Continuous Redundancy Reduction". Prohibited overclaim terms excluded.
- [x] **Standard IEEE Formatting:** Report uses `\documentclass[conference]{IEEEtran}`, standard margins, no non-standard running headers or lines, clean centered page numbering, and compiles to **exactly 4 pages**.
- [x] **Student Identity Standardized:** All files consistently attributed to Khagesh Ranjan (Roll: 24cs2021), Rajiv Gandhi Institute of Petroleum Technology (RGIPT); zero placeholders remain.
- [x] **Licensing & Attribution Explicit:** MIT license explicitly restricted to student-authored code; Barlow Twins objective attributed to Facebook Research (`facebookresearch/barlowtwins`) and ResNet-18 to `torchvision`.
- [x] **Pruned Scope & Limitations:** All experimental research branches cleanly removed from submission. Prospective work framed as general future research (multi-seed validation, longer schedules, CIFAR-100/TinyImageNet, reliable weighting).
- [x] **No Leaked Windows Paths:** Verified that `notebooks/DL070_BarlowTwins_Colab.ipynb` contains zero local Windows paths (`C:\`, `D:\`) and executes in any POSIX/cloud environment.
- [x] **Page Count Adherence:** Verified that `DL070_BarlowTwins_Report.pdf` has **exactly 4 pages** (adheres to the 4–6 page requirement).
