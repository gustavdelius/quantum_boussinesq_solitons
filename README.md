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
| `paper/` | Final paper and its review/checking material |
| `scripts_inventory.md` | Verification-script inventory, coverage, and recorded failures |
| `literature/` | Full texts used (local only, **git-ignored**, so PDFs are not redistributed) |

**Document format.** Phase 1 and Phase 2 are written in **LaTeX** (compiled to PDF), suitable for heavy mathematics. Computations are in **Python/SymPy** scripts that the documents reference.

## Status

- **One-body vacuum matching: the leading discrepancy is explained (4 November 2026).** [Scheme package](phase2/onebody_scheme.pdf), [audit](phase2/audit_onebody_scheme.pdf). **(c), conditional on the continued trace:** Toda has $C_T=1/(2\pi)$; the continuum value $2/\pi$ uses sound shift $9/(4\pi)$ against Toda's $3/(4\pi)$. Vacuum elasticity alone fixes $\Delta\mathcal H=-3\hbar_Bu^2/(2\pi)$, giving exactly $1/(2\pi)$; the sound-subtracted slope $C-s_1=-1/(4\pi)$ already agrees. **(d):** full quantum equivalence, the physical bad-equation determinant and uniform limits remain open; no good-equation prescription is forced by Toda.

- **High-precision Toda one-body coefficient.** [Precision package](phase2/onebody_toda_precision.pdf), [one-body audit](phase2/audit_onebody.pdf), [controlling scheme audit](phase2/audit_onebody_scheme.pdf). **(c):** an unconstrained numerical fit gives $0.1591549414\pm8.24\times10^{-6}$ (deterministic envelope, about $5\times10^{-5}$ relative), and a separate analytic edge calculation gives $1/(2\pi)$. **(d):** uniform fixed-momentum/edge limit interchange remains unproved.

- **Initial one-body dispersion test.** [Package](phase2/onebody_dispersion_test.pdf), [audit](phase2/audit_onebody.pdf), [later correction](phase2/audit_onebody_scheme.pdf). **(c):** six dictionary normalisations introduce no missing factor of four; the earlier zero-continuum-sound-shift claim is refuted. `audit_onebody.py` still fails its line-383 tolerance: crude three-point extrapolation gives $2.25\times10^{-4}$, although the limiting identity is confirmed independently to below $10^{-45}$.

- **Quantum frame dictionary.** [Package](phase2/frame_dictionary.pdf), [audit](phase2/audit_frame_dictionary.pdf). **(c):** endpoint, Jacobian and pressure-response identities and the mixed classical-frame remainder survive; cluster separability and the Krein PDE exclude the displayed antisymmetric addition as a physical Toda frame ambiguity. **(d):** identifying the allowed endpoint responses with continuum collective operators and the physical two-body phase remains open.

- **Projected one-loop determinant and collective measure.** [Package](phase2/det_projected_oneloop.pdf), [audit](phase2/audit_det_projected.pdf). **(c):** instantaneous geometry and intrinsic kinetic-tail divergence are confirmed; fixed-momentum $F=-L_1$ is invariant between the two charge pairs. Its trace value remains conditional on the cutoff endpoint and finite subtraction. **(d):** the boundary-compatible functional determinant and absolute phase remain open. Four tautological check groups are replaced by independent tests in the audit.

- **S-matrix candidate constraints.** [Package](phase2/smx_candidates.pdf), [audit](phase2/audit_smx.pdf). **(c):** classical/Gamma algebra and bad candidate breather poles survive; universal exclusions and several printed ring, crossing and small-amplitude formulas fail. Under local analytic and loop assumptions, one antisymmetric overtaking function remains the continuum's residual $O(\hbar^0)$ two-body unknown, of the same kind as the frame obstruction. **(d):** equality of functions needs the quantum map; the residual is not a freely selectable physical convention on either side. An exact S-matrix and quantum ring holonomy remain unestablished.

- **Defect charges, boundary renormalisation and fusion.** [Package](phase2/defq_charges_fusion.pdf), [audit](phase2/audit_defq.pdf). **(c):** classical charges, KP-edge factorisation, momentum trace, linear UV divergence and admissible growing-mode counts survive. Scattering uses $s=\sigma_{\rm vac}$; the spectrum uses $\eta\sigma_{\rm vac}$. The purported absorption test and printed energy sign, asymptotic phase, threshold rate and pole argument fail. **(d):** sampled holonomies do not prove a universal physical fusion no-go.

- **Consolidated corrections and verification coverage.** [Errata](phase2/errata.pdf), [reference audit](phase0/reference_audit.md), [script inventory](scripts_inventory.md), and [paper review](paper/review_draft1.pdf). **(c):** the addendum records subsequent failed assertions, tautological checks, the repaired disabled Hirota assertion, and candidate/defect refutations. Passing scripts certify their asserted checks, not every printed formula.

- **Phase 2i, conditional Wiener–Hopf asymptotics and KP bookkeeping (4 November 2026).** See [package](phase2/toda_wh_and_kp_synthesis.pdf), [audit](phase2/audit_phase2i.pdf), and [errata](phase2/errata.pdf). **(c), conditional:** corrected-transform reversion and independent numerical inversion support the stated edge coefficients; the leading residual and two-loop coefficient are verified evaluations of formal integrals. The printed Fourier symbol, momentum normalization and scale, inverse series, double transform, Darboux delay, kernel necessity theorem, and four-term reciprocity product require correction. The quoted saved-data agreement uses a fixed leading coefficient and one extrapolation basis; fit sensitivity exceeds the quoted precision. Explicit orientation offsets and crossing identities are algebraically checked. **(d):** uniform finite-sea edge asymptotics, the physical quantum frame, absolute phase and fusion constants, the projected determinant, and a physical all-loop continuum amplitude remain open. Phase 2i does not resolve all earlier audits.

- **Follow-up dictionary and projected-measure audits (4 November 2026).** [Toda dictionary](phase2/audit_toda_dictionary.pdf) and [projected collective measure](phase2/audit_projected_measure.pdf), with independent `audit_toda_dictionary.py` and `audit_projected_measure.py` checks. **(c):** exact Fredholm identities fix the dressed edge momentum and the symmetric phase in terms of endpoint responses; a converged finite-ħ test detects a nonseparable discrepancy in the retained classical frame. **(c):** potential-tail cross terms cancel from the collective symplectic volume, verified through an exact two-soliton collision, but the explicit projector does not preserve fixed potential endpoints. **(d):** the physical quantum frame, its edge-limit correction, and the full boundary-compatible field determinant remain open. **(c):** the master report incorporates many audit qualifications; remaining action-sign, defect-root, ratio, script-name and bibliographic corrections are recorded in [errata](phase2/errata.pdf). The historical write-ups are preserved. These listed master-report corrections are applied in the present consolidation.

- **Independent quantum-foundations audits (4 November 2026).** Notes on [charges and absorption](phase2/audit_charges_action.pdf), [determinant and phase constants](phase2/audit_determinant_foundations.pdf), and [defect spectrum and Toda frame](phase2/audit_frames_defects.pdf); independent checks `phase2/scripts/audit_charges_action.py`, `audit_determinant_foundations.py`, and `audit_frames_defects.py`. **(c), conditional:** the regulated co-moving mode sum reproduces the proposed charge Lagrangian when its cutoff endpoint is retained; direct potential-field quadrature confirms the phonon density identity. **(c):** the displayed defect sewing determinant misses a factor of 16, and the growing-mode formula misses an admissible-branch change. **(d):** the complete projected determinant, defect boundary renormalisation, physical fusion constants, and quantum frame dictionary remain unresolved. The direct Toda expansion can proceed independently with these qualifications.

- **Independent Phase 2h audit (4 November 2026).** [`phase2/independent_phase2h_audit.pdf`](phase2/independent_phase2h_audit.pdf); checks and results `phase2/scripts/audit_phase2h_*`. **(c):** independent phonon quadrature finds orientation-dependent π discrepancies in the master-function identification; raw-phase antisymmetry and the separated head-on asymptotics need correction. Saved Toda extrapolation alternatives reverse the quoted subleading coefficient signs, and the proposed momentum shift generates a classical `(log Y)^2/Y` Taylor term. **(d):** the exact leading Toda match and the physical all-loop kernel remain unestablished; this audit qualifies the earlier Phase 2h claims below.

- **Phase 0, literature search recorded.** **(c):** the project search found no prior Boussinesq soliton S-matrix. The recorded prior one-body semiclassical breather quantisation is Bogdan–Charkina, arXiv:2502.14695, for the bad equation. See [literature map](phase0/literature_map.md) and [errata](phase2/errata.pdf) for bibliographic corrections and reading limitations. This search result is not a proof that no other prior work exists.
