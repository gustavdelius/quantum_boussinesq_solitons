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
- **Phase 2b, Route D (quantum Toda chain).** Write-up in [`phase2/toda_route.pdf`](phase2/toda_route.pdf); scripts `phase2/scripts/toda_*.py`.
  - The classical limit of the Toda Bethe ansatz reproduces the Toda soliton sector exactly, including the topological cross terms.
  - The Toda label-frame phase has the closed form δ_T = (p₁−p₂) log A_T − (Π₁M₂−Π₂M₁).
  - In the Boussinesq scaling limit δ_T → 3b·δ_bad, for overtaking and head-on collisions.
  - O(ħ) quantum corrections were computed at finite ħ.
  - Limitation: Toda reaches only small-amplitude Boussinesq solitons.
- **Phase 2c, candidate exact S-matrix (bad equation).** [`phase2/smatrix_ansatz.pdf`](phase2/smatrix_ansatz.pdf); scripts `phase2/scripts/smatrix_*.py`. A guess, (d): Γ(1±iη)/Γ(1∓iη) in η = (2/9)(ℓ₁−ℓ₂)/ħ, times a power of the root cross-ratio and the topological factor. It passes the semiclassical limit (up to ±π/2), unitarity, crossing, and has poles exactly at the Bogdan–Charkina levels. Gamma functions of root differences and mirror forms F(β₁−β₂)G(β₁+β₂) are excluded. O(ħ⁰) ambiguities (CDD, ħ renormalisation) remain.
- **Phase 2c, soliton–defect semiclassics (good equation, Corrigan–Zambon defect).** Write-up in [`phase2/defect_semiclassics.pdf`](phase2/defect_semiclassics.pdf); scripts `corrigan-zambon/defect_*.py`. Transmission phase δ_D = (2/9)(ℓ − ℓ(q)) log z − (2/3)(ΠM_D − Π_D M), where the defect carries one KP root q = −s/(2√3). The defect adds no cross terms. Two-soliton–defect scattering is symplectic and factorises. Defect vacua are linearly stable iff ησ < −1 (one per KP triple), so excited defects are unstable. Reflection and two-soliton decay are not covered.
- **Phase 2d, Fermi-edge double-scaling limit of quantum Toda (exact small-amplitude scattering).** [`phase2/toda_edge.pdf`](phase2/toda_edge.pdf); scripts `phase2/scripts/toda_edge_*.py`. Head-on: exact phase = δ/ħ (the ±π/2 is a convention here). Overtaking: the Phase 2c candidate matches the exact phase to ≲10⁻³ for η ≳ 2.5, but a genuine two-body residual of up to ~0.1 remains near threshold. So the candidate is not exact.
- **Phase 2e, one-loop two-soliton scattering phase (good equation).** Write-up in [`phase2/oneloop_phase.pdf`](phase2/oneloop_phase.pdf); scripts `phase2/scripts/oneloop_good_*.py`. Phonon scattering on the one-soliton is reflectionless and unitary with a three-root phase shift; Levinson's theorem counts the zero mode for α < π/3 and an extra heavy-branch instability mode for α > π/3. On the two-soliton background, the Faddeev–Korepin one-loop phase evaluates in closed form in terms of Legendre's χ₂ function. Being a non-constant dilogarithmic function of (α₁, α₂), it rules out both the continued Gamma candidate (A) and the pure exponential (B) as exact, while reproducing (A)'s −π/2 constant as α₁ → α₂ (from the q = 0 zero-mode jump in the overtaking channel) and matching (B) and the quantum Toda limit in the small-amplitude head-on corner.
- **Phase 2f, Toda/one-loop matching audit.** [`phase2/matching.pdf`](phase2/matching.pdf); scripts and data `phase2/scripts/matching_*`. **(c), conditional on the Phase 2e one-loop formulas:** the full good-equation correction at fixed physical momentum grows logarithmically at small amplitude in both channels; the momentum expansion requires α²/ħ ≫ |log α|. Thus the fixed-α phonon limit in Phase 2e does not establish a match to fixed-physical-momentum Toda scattering. The note also corrects the sign in Phase 2e's claimed cancellation to a pure χ₂ function. Independent Toda scans separate mesh, edge-extrapolation, and large-y fit errors while retaining the existing classical frame conversion. A common quantum momentum/frame dictionary remains unresolved **(d)**; no new exact S-matrix is claimed.
- **Phase 2g, one-loop soliton–defect transmission phase and KP-triangle fusion (good equation, Route E step 2).** Write-up in [`phase2/oneloop_defect_phase.pdf`](phase2/oneloop_defect_phase.pdf); scripts `phase2/scripts/oneloop_defect_*.py`. Phonon scattering on $N$-soliton + Corrigan–Zambon defect backgrounds is reflectionless with factorised transmission amplitude; the defect phase-shift derivative is a 2-root $(-\eta,+\eta)$ dipole over the elementary kernel $\Theta'(q;\psi)$ whose chord difference reproduces the 3-root $(1,1,-2)$ soliton kernel. The $4\times 4$ defect-vacuum localised-mode determinant factors into two quadratics with closed-form eigenvalues $\omega_{1,2} = i[\eta\sigma\sqrt{4-\sigma^2} \pm \sqrt{3}(2-\sigma^2)]/8$. Because the static defect has no translational zero mode ($\theta_D^{\mathrm{phys}}(0^\pm)=0$), the raw and antisymmetrised Faddeev–Korepin integrals coincide ($I_{SD}=J_{SD}^{\mathrm{phys}}$) and evaluate in closed form as a $3\times 2$ Legendre $\chi_2$ sum. One-loop defect vacuum shifts $(\Delta E_D^{\mathrm{ren}},\Delta P_D)$ balance $(\Delta E_1^{\mathrm{ren}},\Delta P_1)$ in soliton absorption, and one-loop defect fusion $I_{SD}(3;q_{i+1},\eta)-I_{SD}(3;q_i,\eta)=\rho_i I_{SS}(3,S_i)$ holds on every KP edge and closes around every KP triangle ($\sum_i \rho_i I_{SS}=0$), whereas the $+\frac{\pi}{2}\delta_{\epsilon_3,\epsilon_{S_i}}$ constant of $J_{SS}$ fails to close by $\pm\pi/2$.
