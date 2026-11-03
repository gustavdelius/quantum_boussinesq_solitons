# Quantum solitons of the Boussinesq equation

This is a research project on the quantisation of the soliton solutions of the Boussinesq equation, including their two-body quantum scattering data (an S-matrix or its non-relativistic analogue). If such data can be obtained, the aim is to derive them.

## Ground rules

Every nontrivial claim is labelled:
- **(a)** proved in the literature;
- **(b)** conjectured or heuristic in the literature;
- **(c)** derived in this project;
- **(d)** a guess.

References carry arXiv IDs or DOIs. Each note says whether the full text or only an abstract was read. Calculations are checked by computer algebra or numerics, and the checks that were run are recorded, including failures.

## Fixed convention

Unless stated otherwise, everywhere in this project:

$$u_{tt} = u_{xx} - u_{xxxx} - 3\,(u^2)_{xx}, \qquad u = 2(\log\tau)_{xx},\qquad (D_t^2 - D_x^2 + D_x^4)\,\tau\cdot\tau = 0 .$$

- This is the **"good"** (well-posed) Boussinesq equation. Its linear dispersion is ω² = k² + k⁴, real for all real k.
- A one-soliton has τ = 1 + e^{kx − ωt} with ω² = k² − k⁴, 0 < k < 1. It is a positive sech² pulse of speed √(1−k²) < 1, so solitons are subsonic.
- The sign of the nonlinear term is conventional (u → −u). It was checked symbolically on the one-soliton by `phase0/checks/convention_check.py`.
- The **"bad"** equation is u_tt = u_xx + u_xxxx + 3(u²)_xx. Its one-soliton has ω² = k² + k⁴ and is supersonic.

Phase 1 may need other forms (potential form, the first-order W₃ / Lax system L = ∂³ + u∂ + w). Each will be related to this one explicitly.

## Layout

| Path | Contents |
|---|---|
| `phase0/literature_map.md` | Phase 0: annotated map of the field and verdict on prior results |
| `phase1/` | Phase 1: review of classical Boussinesq solitons (`review.tex` / `review.pdf`), `scripts/`, `figures/` |
| `phase2/` | Phase 2: report on quantisation routes (`report.tex` / `report.pdf`), `scripts/` |
| `literature/` | Full texts used (local only, **git-ignored**, so PDFs are not redistributed) |

**Document format.** Phase 1 and Phase 2 are written in **LaTeX** (compiled to PDF), suitable for heavy mathematics. Computations are in **Python/SymPy** scripts that the documents reference.

## Status

- **Phase 0, done.** No prior soliton S-matrix for Boussinesq was found. The only prior soliton quantisation is the one-body Bohr–Sommerfeld quantisation of the breather of the ill-posed equation (Bogdan–Charkina, arXiv:2502.14695). See the map for details and caveats.
- **Phase 1, done.** The review is [`phase1/review.pdf`](phase1/review.pdf), source `phase1/review.tex`. The verification scripts are in `phase1/scripts/`; each ends with `ALL CHECKS PASSED`. Rebuild with
  `cd phase1 && pdflatex review.tex`, and run the checks with `cd phase1/scripts && for f in *.py; do python3 $f; done`.
- **Phase 2, first pass done (bad equation, chosen by the PI).** The report is [`phase2/report.pdf`](phase2/report.pdf), with checks in `phase2/scripts/`.
  - No exact S-matrix was obtained.
  - Main result: a closed-form semiclassical two-soliton phase δ = (2/9)(ℓ₁−ℓ₂) log A₁₂ + (2/3)M₁M₂(v₁−v₂). It rests on exact symplecticity of scattering once topological cross terms are included, and it matches the Bogdan–Charkina breather action.
  - The report ends with a ranked list of next steps.
