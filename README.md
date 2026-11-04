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

- **Phase 2i, Wiener–Hopf solution of the Toda Fermi-edge limit, two-loop $O(1/Y)$ asymptotics, and KP master-function synthesis (4 November 2026).** [`phase2/toda_wh_and_kp_synthesis.pdf`](phase2/toda_wh_and_kp_synthesis.pdf); script `phase2/scripts/toda_wh_and_kp_synthesis.py`. **(c):** exact Wiener–Hopf factorisation of the Toda Bethe integral equation on the semi-infinite Fermi sea proves $\delta u(y) = -\frac{\ln y}{2\pi} + \frac{1-\ln(8\pi)}{2\pi} + (\frac{1}{48}-\frac{1}{4\pi^2})y^{-1} + \frac{4+6\zeta(3)-\pi^2}{96\pi^3}y^{-2} + O(y^{-3})$ with vanishing $(\ln y)/y$, $(\ln^2 y)/y^2$, and $(\ln y)/y^2$ coefficients ($d_1=e_2=e_1=0$). **(c):** Krein convolution proves the balanced 4-term coboundary cancellation of all $\ln Y$ and $c_{\mathrm{edge}}$ terms in $\Phi_q(u(Y),u(r^2 Y))$, derives $\lim_{Y\to\infty} R_0(Y,r) = -\frac{4}{\pi}(\chi_2(r)-r)$ analytically, and yields the closed-form two-loop coefficient $c_1(r) = -\frac{r^3}{6(1-r^2)} - 2d_0(\frac{1-r^2}{r^2}\ln\frac{1-r}{1+r} + \frac{2}{r})$, which matches the saved Phase 2f Toda data to $<2\times 10^{-4}$ once the finite-$h$ classical lattice dispersion phase $\delta_T(\kappa_1,\kappa_2)/h$ is subtracted. **(c):** replacing $K_L(z)$ by the odd, inversion-antisymmetric kernel $\widetilde{K}_L(z) = \frac{K_L(z)-K_L(1/z)}{2} = K_L(z) + \frac{\pi}{8}\operatorname{sgn}(\operatorname{Re} z)$ resolves the $\epsilon_1=-1$ orientation discrepancies, gives exact crossing symmetry and Darboux-shifted defect fusion, and continues holomorphically to the breather locus. **(d):** the full fourth-order projected fluctuation determinant, zero-mode collective measure, defect UV counterterm, and all-loop continuum kernel outside the Toda corner remain open.

- **Follow-up dictionary and projected-measure audits (4 November 2026).** [Toda dictionary](phase2/audit_toda_dictionary.pdf) and [projected collective measure](phase2/audit_projected_measure.pdf), with independent `audit_toda_dictionary.py` and `audit_projected_measure.py` checks. **(c):** exact Fredholm identities fix the dressed edge momentum and the symmetric phase in terms of endpoint responses; a converged finite-ħ test detects a nonseparable discrepancy in the retained classical frame. **(c):** potential-tail cross terms cancel from the collective symplectic volume, verified through an exact two-soliton collision, but the explicit projector does not preserve fixed potential endpoints. **(d):** the physical quantum frame, its edge-limit correction, and the full boundary-compatible field determinant remain open. The [master report](phase2/report.pdf) now reflects the audits throughout its abstract, route summaries, consistency table, and priorities.

- **Independent quantum-foundations audits (4 November 2026).** Notes on [charges and absorption](phase2/audit_charges_action.pdf), [determinant and phase constants](phase2/audit_determinant_foundations.pdf), and [defect spectrum and Toda frame](phase2/audit_frames_defects.pdf); independent checks `phase2/scripts/audit_charges_action.py`, `audit_determinant_foundations.py`, and `audit_frames_defects.py`. **(c), conditional:** the regulated co-moving mode sum reproduces the proposed charge Lagrangian when its cutoff endpoint is retained; direct potential-field quadrature confirms the phonon density identity. **(c):** the displayed defect sewing determinant misses a factor of 16, and the growing-mode formula misses an admissible-branch change. **(d):** the complete projected determinant, defect boundary renormalisation, physical fusion constants, and quantum frame dictionary remain unresolved. The direct Toda expansion can proceed independently with these qualifications.

- **Independent Phase 2h audit (4 November 2026).** [`phase2/independent_phase2h_audit.pdf`](phase2/independent_phase2h_audit.pdf); checks and results `phase2/scripts/audit_phase2h_*`. **(c):** independent phonon quadrature finds orientation-dependent π discrepancies in the master-function identification; raw-phase antisymmetry and the separated head-on asymptotics need correction. Saved Toda extrapolation alternatives reverse the quoted subleading coefficient signs, and the proposed momentum shift generates a classical `(log Y)^2/Y` Taylor term. **(d):** the exact leading Toda match and the physical all-loop kernel remain unestablished; this audit qualifies the earlier Phase 2h claims below.

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
- **Phase 2d, numerical Fermi-edge double-scaling limit of quantum Toda.** [`phase2/toda_edge.pdf`](phase2/toda_edge.pdf); scripts `phase2/scripts/toda_edge_*.py`. **(c), within the implemented classical frame conversion:** head-on data approach the classical phase, while the overtaking data disagree with the Phase 2c Gamma candidate near threshold. **(d):** interpreting these comparisons as exact continuum quantum scattering requires the common quantum momentum/frame dictionary examined in the follow-up audits; the original write-up itself retains the frame-conversion caveat.
- **Phase 2e, continuum one-loop phase integrals (good equation).** [`phase2/oneloop_phase.pdf`](phase2/oneloop_phase.pdf); scripts `phase2/scripts/oneloop_good_*.py`. **(c):** reflectionless tau-function phonon modes, explicit dilogarithmic continuum integrals, and independently checked potential-field spectral density. **(d):** identification with the complete physical one-loop S-matrix requires a projected determinant, boundary domain, and absolute phase normalisation. Raw exchange constants depend on orientation; candidate exclusions and Toda matching remain conditional. See the independent determinant audits and the revised master report.
- **Phase 2f, Toda/one-loop matching audit.** [`phase2/matching.pdf`](phase2/matching.pdf); scripts and data `phase2/scripts/matching_*`. **(c), conditional on the Phase 2e one-loop formulas:** the full good-equation correction at fixed physical momentum grows logarithmically at small amplitude in both channels; the momentum expansion requires α²/ħ ≫ |log α|. Thus the fixed-α phonon limit in Phase 2e does not establish a match to fixed-physical-momentum Toda scattering. The note also corrects the sign in Phase 2e's claimed cancellation to a pure χ₂ function. Independent Toda scans separate mesh, edge-extrapolation, and large-y fit errors while retaining the existing classical frame conversion. A common quantum momentum/frame dictionary remains unresolved **(d)**; no new exact S-matrix is claimed.
- **Phase 2g, continuum soliton–defect phase and formal KP fusion.** [`phase2/oneloop_defect_phase.pdf`](phase2/oneloop_defect_phase.pdf); scripts `phase2/scripts/oneloop_defect_*.py`. **(c):** factorised phonon transmission, dipole/chord identities, and closed continuum integrals. Independent audits correct a factor of sixteen in the displayed sewing determinant and a missed growing-mode branch. **(d):** physical quantum absorption and fusion still require boundary renormalisation, localized conserved charges, and normalized admissible defect states; formal triangle closure does not determine the scattering constant.
- **Phase 2h, proposed charge/frame synthesis and KP master function.** [`phase2/charge_frame_master.pdf`](phase2/charge_frame_master.pdf); scripts `phase2/scripts/charge_and_frame_audit.py` and `kp_master_and_toda_edge.py`. **(c), conditional:** the regulated co-moving mode sum reproduces the proposed charge Lagrangian and its momentum derivative; abstract KP chord closure holds. **(c):** independent checks find orientation-dependent discrepancies in the physical master identification, unresolved subleading Toda fits, and a logarithmic classical Taylor remainder. **(d):** the common quantum dictionary, the proposed exact leading Toda residual, and an all-loop physical kernel remain unestablished. The revised [`phase2/report.pdf`](phase2/report.pdf) replaces the earlier claim of complete reconciliation with these qualifications.
