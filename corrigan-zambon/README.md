# Do Corrigan–Zambon defects exist for the bad Boussinesq equation?

**Question.** Corrigan–Zambon (arXiv:2306.03772, *J. Phys. A* 56 (2023)) build integrable defects for the *good* Boussinesq equation (κ² > 0). Does their construction carry over to the *bad* equation (κ² < 0)? If it does not, quantising their defects forces us to quantise the good equation.

**Answer.**
- **With the nonlinear term (λ ≠ 0): no.** Within their type-I ansatz, there is no real defect for κ² < 0. The defect Lagrangian necessarily has coefficients α, β ∈ iℝ.
- **For the linear equation (λ = 0): yes.** Real solutions exist.
- **Status.** The construction itself is (a), from their paper. The continuation to κ² < 0 and the obstruction are (c), derived and verified in `defect_check.py`.

## Why they chose the good equation

The choice is for modelling reasons, not mathematical ones. In §2 they assume κ real and positive, being "the form … most relevant to the investigation of pulse propagation in nerve fibres", citing Heimburg–Jackson (their ref. [11]). They note that for κ² < 0 solitons are real only for |ν| > c. They give no other justification.

## Set-up (their conventions, κ² → K)

Bulk Lagrangian (their 2.1):

$$\mathcal L=\frac{u_t^2}{2c^2}-\frac{u_x^2}{2}-\frac{\lambda}{3}u_x^3-\frac{K}{2}u_{xx}^2 .$$

Here K > 0 is the good equation and K < 0 the bad one. The field is u for x < x₀ and v for x > x₀. The defect term is the type-I ansatz

$$D=u_tF+v_tG+H\qquad\text{at }x_0,$$

with F, G, H functions of u, uₓ, v, vₓ.

**Requirement.** The total momentum P − uₓF − vₓG + P_D must be conserved for some P_D(u, uₓ, v, vₓ). This is what makes the defect "integrable" (their 3.10–3.16).

## Results (all in `defect_check.py`)

Everything is carried out with K a free symbol, starting from the variation of the action. In particular, the sewing conditions and the momentum flux are rederived, not copied from the paper.

1. **Terms quadratic in time derivatives.**
   - Take F = a₁uₓ + b₁vₓ + f and G = b₂uₓ + a₂vₓ + g, with constant coefficients.
   - Exactness of P_D forces b₂ = b₁ (their 3.15). Then a₂ = a₁ =: α and b₁ =: β, with

   $$\beta^2-\alpha^2=K/c^2 \qquad\text{(their 3.22)}.$$

   This is real for either sign of K. (c), reproducing (a)
2. **Terms cubic in space derivatives.**
   - H has a general quadratic part with coefficient functions h₁₁, h₁₂, h₂₂(u, v).
   - These coefficients are forced into the form σ(βuₓ² + 2αuₓvₓ + βvₓ²)/2.
   - For λ ≠ 0 the only solution is

   $$\beta=2\alpha,\qquad \partial_u\sigma=-\partial_v\sigma=\frac{\lambda}{3\alpha}\qquad\text{(their 3.35–3.37)}.$$

   Combined with item 1 this gives **α² = K/(3c²)**. (c), reproducing (a)
3. **Sufficiency.**
   - Their complete solution (σ, ρ, τ, and f, g with f_v − g_u = σ/c²) is written with α a free complex parameter and K := 3c²α².
   - With it, the momentum balance is exactly dP_D/dt identically: every non-exact piece vanishes and the 1-form for dP_D is closed.
   - So the construction is algebraic in α and continues to any complex α. (c)
4. **Reality for K < 0.**
   - Write K = −κ². Then α = ±iκ/(√3 c) and β = 2α are purely imaginary.
   - α is the coefficient of u_t u_x in D. The functions f and g depend only on u and v, and the gauge freedom f → f + Ω_u, g → g + Ω_v cannot change that coefficient.
   - Hence D is not real for real fields: **no real defect of this family exists for the bad equation when λ ≠ 0**. (c)
5. **λ = 0.** Here β = 2α is not forced, and β² − α² = −κ²/c² has real solutions (e.g. α = κ/c, β = 0). So the obstruction comes from the nonlinearity, not from the dispersion sign alone. (c)

## Caveats and what is not excluded

- **Constancy of α and β.** I assumed α and β constant, as the paper derives in 3.28–3.30. I did not rederive that step. If non-constant coefficients were possible, the analysis would change. (a, from the paper)
- **Type-II defects.** Defects with their own degree of freedom are not covered. In sine-Gordon these exist alongside type I, so a real type-II defect for the bad equation remains possible. (d)
- **Complexified theories.** One could allow a non-real defect action, for example PT-symmetric. That goes beyond the paper's real-action setting. (d)
- **Different media.** Different bulk constants on the two sides (mentioned in their §3) were not explored. In particular we did not explore a defect joining a good region to a bad region.

## Consequence for the project

Quantising the Corrigan–Zambon defects, as constructed, requires the **good** bulk theory. With the bad bulk theory, one would first have to find a type-II (or otherwise generalised) defect.

This is consistent with the guess in the Phase 1 review that their novel defect phenomena (reflection, decay into solitons) are tied to the resonant, number-changing interactions that only the good equation has. That link remains a guess. (d)

Run: `python3 defect_check.py`, which ends with `ALL CHECKS PASSED`.

## Interactive explorer

`app/defect_explorer.html` is a standalone page. Open it in any browser, or use the published copy at https://claude.ai/artifact/Qw4Skejo6Q4MWmdmVYE4Ts (private by default). It animates the exact soliton–defect solutions of CZ §5: one soliton, two solitons, and merging/splitting solitons.

`verify_defect_solutions.py` checks these solutions against the sewing conditions derived from the action. They hold to 10⁻²⁸–10⁻³¹, or to about 10⁻¹⁵ when evaluated near a pole in the singular z < 0 cases.

The paper's example gauge f = g = −σ²/(2μc²) (its eqs. 3.43/3.47) gives f_v − g_u = 2σ/c² instead of σ/c². This looks like a typo. Only f_v − g_u matters, so the script uses f = ∫σ dv / c², g = 0.
