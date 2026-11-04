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

The [classical review](phase1/review.pdf) gives the potential and third-order Lax formulations in this convention.

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

## Open decisions

Choices that wait on the PI (see `AGENTS.md`). Move an item to "Resolved" with the decision and its commit.

- **Bad-equation determinant prescription (validation item V3).** The one-loop problem on the bad equation is ill-posed on the full phase space and well posed on the Charlier–Lenells class. On 3 November 2026 three regularisations were put to the PI: (A) the Charlier–Lenells-reduced theory with stable phonons |q| < 1 only (recommended, labelled as a choice of quantum theory); (B) the Toda lattice as the definition; (C) continuation from the good equation. No choice has been recorded. The [master report](phase2/report.pdf) (V3) asks for a physical prescription, or a PI choice, before the continued trace is extended. V1 depends on it.

- **Higher-charge and ordering convention for that test (4 November 2026).** **(d)** Provisionally use the first higher residue H₇ modulo lower charges, free-good-vacuum Wick products, and the existing sharp momentum cutoff on contractions with the continuum canonical bracket (recommended). Alternatives: another hierarchy representative or a separately constructed integrable operator regulator. **(c)** The required local charge mixing is displayed explicitly; a finite-mode cutoff and convergence of the continuum charge have not been established. Raised in [the scheme note](phase2/integ_scheme.pdf); originating commit `e8a8635`.

Resolved:

- **Good-equation scheme after the integrability test (4 November 2026, raised in `e8a8635`).** The PI chose option (A), the static reference-vacuum scheme: free-good-vacuum Wick subtraction, e(0)=e′(0)=0 and e″(0)=1, hence λ=9/(4π) (s₁=0, C_g=1/(4π)). The one-loop light–light phase is to be computed in this scheme, keeping the arbitrary-λ response K_λ for conversion ([scheme note](phase2/integ_scheme.pdf)). The higher-charge and ordering convention entry stays open.
- **Target for the good-equation S-matrix (4 November 2026, raised in `5efbdca`).** The PI chose (C) with (A). The deliverable is the complete O(ħ⁰) light–light two-soliton phase in a stated scheme. The finite counterterms are to be fixed by requiring a quantum higher charge to commute with H at O(ħ) (quantum integrability as the scheme condition). An exact S-matrix from an integrable completion (option B) is a long-term programme, not the paper's target.
- **Priority (4 November 2026, `446bf1a`, `26eba1d`).** The good equation is the main thread; the bad equation and quantum Toda serve as validation.
- **±π/2 constant in the continued bad candidate.** The defect-triangle test (`e3328a6`) found a failure by e^{±iπ/2}. The later candidate audit (`phase2/audit_smx.pdf`) refuted the printed crossing offset: direct quadrature gives crossing sum zero, not π/2, and the checker had inserted the offset by hand. **(d):** this probably removes the source of the defect failure, but the defect test has not been rerun without the offset. Candidate selection remains open as next step (5) in the report.

## Status

The PI's priority is quantum scattering on the **good equation**. The bad equation and the quantum Toda chain provide validation tests. The previous README was updated at `adb9116` (12:19 UTC on 4 November 2026) and the report at `446bf1a` (12:30 UTC). The [Phase 2 master report](phase2/report.pdf) now covers the subsequent work through `bfc3944` (17:08 UTC).

The vacuum condition, defect bootstrap and two shift-charge notes report package checks. For the other new Phase 2 results, the linked independent audits give the controlling verdicts.

**Good-equation results**

- **Integrability scheme test.** **(c)** An explicit higher charge and its local one-loop Ward correction leave λ free. Static modulus matching would independently give λ=9/(4π); sound-unit one- and two-body invariants are verified. **(d)** Static matching is recommended provisionally; continuum charge existence and the absolute phase remain open. [Scheme note](phase2/integ_scheme.pdf)
- **Vacuum condition and elastic response.** **(c)** With a finite term $\hbar\lambda\int u^2\,dx$, the conditional soliton trace gives $C_g=\lambda-2/\pi$ and the static sound shift is $s_1=\lambda-9/(4\pi)$. Preserving classical vacuum elasticity would give $\lambda=9/(4\pi)$ and $C_g=1/(4\pi)$. The two-soliton response to $\lambda$ at fixed momenta is antisymmetric in the chamber-reversing Darboux frame. Three-soliton phonon transmission factorises, but clustering leaves $F_{12}=c(P_1M_2-P_2M_1)$ free. **(d)** A physical scheme and the absolute two-body phase remain open. [Vacuum condition](phase2/good_renorm.pdf) · [Two-body calculation](phase2/good_twobody.pdf) · [Two-body audit](phase2/audit_good_twobody.pdf)
- **Light and heavy channels.** **(c)** For every interior heavy soliton there is an on-shell pair of oppositely moving light solitons with the same classical charges. The heavy soliton itself has a localized growing mode. On the light branch, energy and momentum forbid splitting into two light solitons. The formal hierarchy charges telescope across a Y junction, with explicit checks through $H_8$. **(d)** Quantum decay widths and closure of the light scattering sector remain open. [Resonant channels](phase2/resonant_channels.pdf) · [Independent audit](phase2/audit_resonant_channels.pdf)
- **Defect boundary and bootstrap.** **(c)** The canonical energy cycle sum is nonzero on every nondegenerate real KP triangle at $\lambda_g=0$. Allowing the absorption rapidities to move removes the proposed constraint on the bulk coupling. Every such cycle contains a heavy chord, and even its formal edge equations and two-defect consistency impose no constraint on the one coefficient in $F_{12}$. **(d)** Finite quantum boundary terms, residues and physical fusion remain open. [Boundary calculation](phase2/defb_boundary.pdf) · [Boundary audit](phase2/audit_defb_boundary.pdf) · [Bootstrap](phase2/defect_bootstrap.pdf)
- **Shift charge and the quantum scattering states.** **(c)** In the conditional mean-profile calculation, varying the kink height at fixed momentum cancels the apparent leading term from the shift charge alone. An anchored classical current deformation changes the two-body action by $-g(P_1M_2-P_2M_1)$ while preserving one-soliton energy and momentum. A large-background $W_3$ contraction recovers the canonical bracket and Baxter spacing, but the inherited highest-weight vacuum fails positivity. **(d)** The absolute coefficient, normalized Baxter operators, physical soliton states and boundary-compatible determinant remain open. [Shift-charge calculation](phase2/shiftcharge_oneloop.pdf) · [Quantum bridge](phase2/shiftcharge_quantum_bridge.pdf)

**Validation and paper**

- **V1, two-body comparison.** **(c)** The iterated Toda edge limit gives a closed antisymmetric directional coefficient, $1.530527813279\ldots$ at ratio $r=1/2$. The bad-continuum candidate reproduces its rational recoil term. One mixed difference agrees when evaluated with Toda's endpoint identity, but the full comparison retains a ratio-dependent gap. **(d)** V1 remains open without a common quantum momentum and endpoint dictionary and a physical bad-equation determinant. [Toda calculation](phase2/twobody_toda.pdf) · [Toda audit](phase2/audit_twobody_toda.pdf) · [Continuum calculation](phase2/twobody_cont.pdf) · [Continuum audit](phase2/audit_twobody_cont.pdf)
- **V2, order of limits.** **(c)** The logarithmic Taylor term cancels in the matched half-line expansion. Independent inversion supports $c_1(1/2)\simeq-0.021445$ there. Finite-sea fits still change the sign of the subleading coefficient across extrapolation bases, and fixed edge coordinate sends the momentum to zero. **(d)** Uniform interchange with the fixed-momentum limit remains open. [Uniform limits](phase2/uniform_limits.pdf) · [Independent audit](phase2/audit_uniform_limits.pdf)
- **Paper draft 4.** Draft 3 incorporated the audited quantum sections. Draft 4 applies all eleven cross-review corrections, including the four-term Krein source, symplectic and projector signs, one-loop normalization, and the time-dependent collision operator. The later V1 audits, defect bootstrap and shift-charge work are covered in the master report. [Draft 4](paper/boussinesq_soliton_scattering.pdf) · [Draft 3 review](paper/review_draft3.pdf) · [Draft 4 checks](paper/scripts/paper_draft4_checks.py)

### Earlier packages and audits

- **One-body vacuum matching: the leading discrepancy is explained (4 November 2026).** [Scheme package](phase2/onebody_scheme.pdf), [audit](phase2/audit_onebody_scheme.pdf). **(c), conditional on the continued trace:** Toda has $C_T=1/(2\pi)$; the continuum value $2/\pi$ uses sound shift $9/(4\pi)$ against Toda's $3/(4\pi)$. Vacuum elasticity alone fixes $\Delta\mathcal H=-3\hbar_Bu^2/(2\pi)$, giving exactly $1/(2\pi)$; the sound-subtracted slope $C-s_1=-1/(4\pi)$ already agrees. **(d):** full quantum equivalence, the physical bad-equation determinant and uniform limits remain open; no good-equation prescription is forced by Toda.

- **High-precision Toda one-body coefficient.** [Precision package](phase2/onebody_toda_precision.pdf), [one-body audit](phase2/audit_onebody.pdf), [controlling scheme audit](phase2/audit_onebody_scheme.pdf). **(c):** an unconstrained numerical fit gives $0.1591549414\pm8.24\times10^{-6}$ (deterministic envelope, about $5\times10^{-5}$ relative), and a separate analytic edge calculation gives $1/(2\pi)$. **(d):** uniform fixed-momentum/edge limit interchange remains unproved.

- **Initial one-body dispersion test.** [Package](phase2/onebody_dispersion_test.pdf), [audit](phase2/audit_onebody.pdf), [later correction](phase2/audit_onebody_scheme.pdf). **(c):** six dictionary normalisations introduce no missing factor of four; the earlier zero-continuum-sound-shift claim is refuted. `audit_onebody.py` still fails its line-383 tolerance: crude three-point extrapolation gives $2.25\times10^{-4}$, although the limiting identity is confirmed independently to below $10^{-45}$.

- **Quantum frame dictionary.** [Package](phase2/frame_dictionary.pdf), [audit](phase2/audit_frame_dictionary.pdf). **(c):** endpoint, Jacobian and pressure-response identities and the mixed classical-frame remainder survive; cluster separability and the Krein PDE exclude the displayed antisymmetric addition as a physical Toda frame ambiguity. **(d):** identifying the allowed endpoint responses with continuum collective operators and the physical two-body phase remains open.

- **Projected one-loop determinant and collective measure.** [Package](phase2/det_projected_oneloop.pdf), [audit](phase2/audit_det_projected.pdf). **(c):** instantaneous geometry and intrinsic kinetic-tail divergence are confirmed; fixed-momentum $F=-L_1$ is invariant between the two charge pairs. Its trace value remains conditional on the cutoff endpoint and finite subtraction. **(d):** the boundary-compatible functional determinant and absolute phase remain open. Four tautological check groups are replaced by independent tests in the audit.

- **S-matrix candidate constraints.** [Package](phase2/smx_candidates.pdf), [audit](phase2/audit_smx.pdf). **(c):** classical/Gamma algebra and bad candidate breather poles survive; universal exclusions and several printed ring, crossing and small-amplitude formulas fail. Under local analytic and loop assumptions, one antisymmetric overtaking function remains the continuum's residual $O(\hbar^0)$ two-body unknown, of the same kind as the frame obstruction. **(d):** equality of functions needs the quantum map; the residual is not a freely selectable physical convention on either side. An exact S-matrix and quantum ring holonomy remain unestablished.

- **Defect charges, boundary renormalisation and fusion.** [Package](phase2/defq_charges_fusion.pdf), [audit](phase2/audit_defq.pdf). **(c):** classical charges, KP-edge factorisation, momentum trace, linear UV divergence and admissible growing-mode counts survive. Scattering uses $s=\sigma_{\rm vac}$; the spectrum uses $\eta\sigma_{\rm vac}$. The purported absorption test and printed energy sign, asymptotic phase, threshold rate and pole argument fail. **(d):** Physical fusion remains open. The later boundary audit extends the zero-scheme canonical energy result to all nondegenerate real triangles.

- **Consolidated corrections and verification coverage.** [Errata](phase2/errata.pdf), [reference audit](phase0/reference_audit.md), [script inventory](scripts_inventory.md), and [paper review](paper/review_draft1.pdf). **(c):** the addendum records subsequent failed assertions, tautological checks, the repaired disabled Hirota assertion, and candidate/defect refutations. Passing scripts certify their asserted checks, not every printed formula.

- **Phase 2i, conditional Wiener–Hopf asymptotics and KP bookkeeping (4 November 2026).** See [package](phase2/toda_wh_and_kp_synthesis.pdf), [audit](phase2/audit_phase2i.pdf), and [errata](phase2/errata.pdf). **(c), conditional:** corrected-transform reversion and independent numerical inversion support the stated edge coefficients; the leading residual and two-loop coefficient are verified evaluations of formal integrals. The printed Fourier symbol, momentum normalization and scale, inverse series, double transform, Darboux delay, kernel necessity theorem, and four-term reciprocity product require correction. The quoted saved-data agreement uses a fixed leading coefficient and one extrapolation basis; fit sensitivity exceeds the quoted precision. Explicit orientation offsets and crossing identities are algebraically checked. **(d):** uniform finite-sea edge asymptotics, the physical quantum frame, absolute phase and fusion constants, the projected determinant, and a physical all-loop continuum amplitude remain open. Phase 2i does not resolve all earlier audits.

- **Follow-up dictionary and projected-measure audits (4 November 2026).** [Toda dictionary](phase2/audit_toda_dictionary.pdf) and [projected collective measure](phase2/audit_projected_measure.pdf), with independent `audit_toda_dictionary.py` and `audit_projected_measure.py` checks. **(c):** exact Fredholm identities fix the dressed edge momentum and the symmetric phase in terms of endpoint responses; a converged finite-ħ test detects a nonseparable discrepancy in the retained classical frame. **(c):** potential-tail cross terms cancel from the collective symplectic volume, verified through an exact two-soliton collision, but the explicit projector does not preserve fixed potential endpoints. **(d):** the physical quantum frame, its edge-limit correction, and the full boundary-compatible field determinant remain open. **(c):** the master report incorporates the audit qualifications and the action-sign, defect-root, ratio, script-name and bibliographic corrections in the [errata](phase2/errata.pdf).

- **Independent quantum-foundations audits (4 November 2026).** Notes on [charges and absorption](phase2/audit_charges_action.pdf), [determinant and phase constants](phase2/audit_determinant_foundations.pdf), and [defect spectrum and Toda frame](phase2/audit_frames_defects.pdf); independent checks `phase2/scripts/audit_charges_action.py`, `audit_determinant_foundations.py`, and `audit_frames_defects.py`. **(c), conditional:** the regulated co-moving mode sum reproduces the proposed charge Lagrangian when its cutoff endpoint is retained; direct potential-field quadrature confirms the phonon density identity. **(c):** the displayed defect sewing determinant misses a factor of 16, and the growing-mode formula misses an admissible-branch change. **(d):** the complete projected determinant, defect boundary renormalisation, physical fusion constants, and quantum frame dictionary remain unresolved. The direct Toda expansion can proceed independently with these qualifications.

- **Independent Phase 2h audit (4 November 2026).** [`phase2/independent_phase2h_audit.pdf`](phase2/independent_phase2h_audit.pdf); checks and results `phase2/scripts/audit_phase2h_*`. **(c):** independent phonon quadrature finds orientation-dependent π discrepancies in the master-function identification; raw-phase antisymmetry and the separated head-on asymptotics need correction. Saved Toda extrapolation alternatives reverse the quoted subleading coefficient signs, and the proposed momentum shift generates a classical `(log Y)^2/Y` Taylor term. **(d):** the exact leading Toda match and the physical all-loop kernel remain unestablished; this audit qualifies the original Phase 2h claims.

- **Phase 0, literature search recorded.** **(c):** the project search found no prior Boussinesq soliton S-matrix. The recorded prior one-body semiclassical breather quantisation is Bogdan–Charkina, arXiv:2502.14695, for the bad equation. See [literature map](phase0/literature_map.md) and [errata](phase2/errata.pdf) for bibliographic corrections and reading limitations. This search result is not a proof that no other prior work exists.
