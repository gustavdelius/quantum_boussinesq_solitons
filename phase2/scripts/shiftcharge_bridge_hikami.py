"""Independent checks of the W3-to-canonical Boussinesq bridge.

Sources read in full-text passages: Hikami nlin/0102021, sections 1--5 and
appendix B; Bazhanov--Hibberd--Khoroshkin hep-th/0105177, sections 1--3 and
the functional-relation discussion; Belov--Chaltikian hep-th/9303166,
sections 1--5.  The tests distinguish a formal algebra contraction from
a positive quantum representation.  No historical script is imported.
"""

import json
from pathlib import Path
import time

import mpmath as mp
import sympy as sp


START = time.monotonic()
mp.mp.dps = 80
x = sp.symbols("x", real=True)
a = sp.Function("a")(x)
b = sp.Function("b")(x)
u = sp.Function("u")(x)
pi = sp.Function("pi")(x)
U = sp.Function("U")(x)
V = sp.Function("V")(x)
Lambda, hbar = sp.symbols("Lambda hbar", positive=True)
n, c = sp.symbols("n c", nonzero=True, real=True)


def simplify_operator(operator):
    return {
        degree: coefficient
        for degree, raw in operator.items()
        if (coefficient := sp.expand(raw)) != 0
    }


def add_operators(*operators):
    result = {}
    for operator in operators:
        for degree, coefficient in operator.items():
            result[degree] = result.get(degree, 0) + coefficient
    return simplify_operator(result)


def compose(left, right):
    """Compose differential operators with coefficients written on the left."""
    result = {}
    for left_degree, left_coefficient in left.items():
        for right_degree, right_coefficient in right.items():
            for derivatives in range(left_degree + 1):
                degree = left_degree + right_degree - derivatives
                coefficient = (
                    left_coefficient
                    * sp.binomial(left_degree, derivatives)
                    * sp.diff(right_coefficient, x, derivatives)
                )
                result[degree] = result.get(degree, 0) + coefficient
    return simplify_operator(result)


def adjoint(operator):
    result = {}
    for degree, coefficient in operator.items():
        result = add_operators(
            result, compose({degree: (-1) ** degree}, {0: coefficient})
        )
    return result


def apply_operator(operator, function):
    return sp.expand(sum(value * sp.diff(function, x, degree)
                         for degree, value in operator.items()))


def assert_zero(expression):
    assert sp.simplify(sp.expand(expression)) == 0, expression


def assert_same_operator(left, right):
    for degree in set(left) | set(right):
        assert_zero(left.get(degree, 0) - right.get(degree, 0))


# Derive the second GD operator from the free-field Miura transform, rather
# than merely checking a printed operator against its own factorization.
miura_U = a**2 + b**2 + 2 * sp.diff(a, x)
miura_V = (
    sp.Rational(2, 3) * b**3 - 2 * a**2 * b
    - 3 * a * sp.diff(b, x) - sp.diff(a, x) * b - sp.diff(b, x, 2)
) / sp.sqrt(3)
miura_fields = [miura_U, miura_V]
frechet = [[{
    degree: sp.diff(field, sp.diff(current, x, degree))
    for degree in range(3)
} for current in (a, b)] for field in miura_fields]
current_bracket = {1: -sp.Rational(1, 2)}
derived = [[add_operators(*[
    compose(compose(frechet[row][current], current_bracket),
            adjoint(frechet[column][current]))
    for current in range(2)
]) for column in range(2)] for row in range(2)]
J2 = [
    [
        {3: 2, 1: -2 * U, 0: -sp.diff(U, x)},
        {1: -3 * V, 0: -2 * sp.diff(V, x)},
    ],
    [
        {1: -3 * V, 0: -sp.diff(V, x)},
        {
            5: -sp.Rational(1, 6),
            3: sp.Rational(5, 6) * U,
            2: sp.Rational(5, 4) * sp.diff(U, x),
            1: sp.Rational(3, 4) * sp.diff(U, x, 2)
               - sp.Rational(2, 3) * U**2,
            0: sp.diff(U, x, 3) / 6 - sp.Rational(2, 3) * U * sp.diff(U, x),
        },
    ],
]
for row in range(2):
    for column in range(2):
        printed_after_miura = {
            degree: sp.sympify(value).subs({U: miura_U, V: miura_V}).doit()
            for degree, value in J2[row][column].items()
        }
        assert_same_operator(derived[row][column], printed_after_miura)
        assert_same_operator(
            add_operators(J2[row][column], adjoint(J2[column][row])), {}
        )

# Exact PDE and energy dictionary, with tau=sqrt(3)*t.
target_U = sp.Rational(1, 4) - sp.Rational(3, 2) * u
target_V = sp.sqrt(3) * pi / 4
flow_U = -2 * sp.diff(V, x)
flow_V = sp.diff(U, x, 3) / 6 - sp.Rational(2, 3) * U * sp.diff(U, x)
assert_zero(sp.sqrt(3) * flow_U.subs(V, target_V).doit()
            + sp.Rational(3, 2) * sp.diff(pi, x))
assert_zero(sp.sqrt(3) * flow_V.subs(U, target_U).doit()
            - sp.sqrt(3) / 4 * sp.diff(u - 3 * u**2 - sp.diff(u, x, 2), x))
F = U**3 + 9 * V**2 + sp.Rational(3, 4) * sp.diff(U, x)**2
canonical_density = (pi**2 + u**2 + sp.diff(u, x)**2) / 2 - u**3
contracted_density = sp.Rational(8, 27) * (
    F - sp.Rational(3, 16) * U + sp.Rational(1, 32)
)
assert_zero(contracted_density.subs({U: target_U, V: target_V}).doit()
            - canonical_density)
assert_zero(F.subs(V, V + Lambda) - F - 18 * Lambda * V - 9 * Lambda**2)

# The first canonical bracket is induced directly by u=phi_x, {phi,pi}=delta.
canonical_coefficient = -3 * sp.sqrt(3) / 8
assert_zero((-sp.Rational(3, 2)) * sp.sqrt(3) / 4
            - canonical_coefficient)
J1 = [[{}, {1: canonical_coefficient}], [{1: canonical_coefficient}, {}]]
kappa = sp.sqrt(3) / (8 * Lambda)
for row in range(2):
    for column in range(2):
        limit = {
            degree: sp.limit(kappa * sp.sympify(coefficient).subs(V, V + Lambda)
                             .doit(), Lambda, sp.oo)
            for degree, coefficient in J2[row][column].items()
        }
        assert_same_operator(limit, J1[row][column])
gradient = [
    sp.Rational(8, 27) * (3 * U**2 - sp.Rational(3, 2) * sp.diff(U, x, 2)
                         - sp.Rational(3, 16)),
    sp.Rational(16, 3) * V,
]
assert_zero(sum(apply_operator(J1[0][j], gradient[j]) for j in range(2))
            - sp.sqrt(3) * flow_U)
assert_zero(sum(apply_operator(J1[1][j], gradient[j]) for j in range(2))
            - sp.sqrt(3) * flow_V)

# Same-vacuum local-map and real Fourier-inertia diagnostics. They apply to
# invertible maps at a fixed background, and explicitly do not exclude the
# singular large-V contraction above.
k, U0, V0, D = sp.symbols("k U0 V0 D", real=True)
fourier_hermitian = k * sp.Matrix([
    [2 * (k**2 + U0), 3 * V0],
    [3 * V0, (k**2 + U0) * (k**2 + 4 * U0) / 6],
])
assert_zero(fourier_hermitian.det()
            - k**2 * ((k**2 + U0)**2 * (k**2 + 4 * U0) / 3 - 9 * V0**2))
vacuum_J2 = sp.diag(2 * D * (D**2 - sp.Rational(1, 4)),
                     -D * (D**2 - 1) * (D**2 - sp.Rational(1, 4)) / 6)
vacuum_J1 = sp.Matrix([[0, canonical_coefficient * D],
                       [canonical_coefficient * D, 0]])
determinant_ratio = sp.factor(vacuum_J2.det() / vacuum_J1.det())
assert_zero(determinant_ratio
            - sp.Rational(64, 81) * (D**2 - sp.Rational(1, 4))**2 * (D**2 - 1))
assert sp.Poly(determinant_ratio, D).degree() == 6
assert_zero(fourier_hermitian.det().subs({U0: sp.Rational(1, 4), V0: 0})
            - k**2 * (k**2 + sp.Rational(1, 4))**2 * (k**2 + 1) / 3)
constant_a, constant_b, root = sp.symbols("constant_a constant_b root", real=True)
constant_U = constant_a**2 + constant_b**2
constant_V = (sp.Rational(2, 3) * constant_b**3
              - 2 * constant_a**2 * constant_b) / sp.sqrt(3)
assert_zero(4 * constant_U**3 - 27 * constant_V**2
            - 4 * constant_a**2 * (constant_a**2 - 3 * constant_b**2)**2)
assert_zero(sp.discriminant(root**3 - U0 * root + V0, root)
            - (4 * U0**3 - 27 * V0**2))

# Exact mode algebra. U_n=24 L_n/c-delta_n0 and
# V_n=W_n/[i(-c/24)^(3/2)]-Lambda delta_n0.
chosen_c = -128 * sp.sqrt(3) * sp.pi * Lambda / hbar
cross_central = 72 * n * Lambda / c
canonical_cross = -3 * sp.sqrt(3) * hbar * n / (16 * sp.pi)
assert_zero(cross_central.subs(c, chosen_c) - canonical_cross)
UU_gram = 48 * n * (n**2 + U0) / c
delta2 = c * (U0 + 1) / 24
b_squared = 16 / (22 + 5 * c)
WW_gram = (
    sp.Rational(2, 3) * n * (delta2**2 + delta2 / 5)
    + n * (5 * n**2 - 8) * delta2 / (45 * b_squared)
    + c * n * (n**2 - 4) * (n**2 - 1) / (1080 * b_squared)
)
VV_gram = sp.factor((24 / c)**3 * WW_gram)
assert_zero(sp.limit(c * VV_gram, c, -sp.oo)
            - 4 * n * (n**2 + U0) * (n**2 + 4 * U0))
assert_zero(sp.limit(UU_gram.subs(c, chosen_c), Lambda, sp.oo))
assert_zero(sp.limit(VV_gram.subs(c, chosen_c), Lambda, sp.oo))
limiting_gram = sp.Matrix([[0, canonical_cross], [canonical_cross, 0]])
assert_zero(limiting_gram.det() + canonical_cross**2)

# The quantum density from BHK Eq2.13 in rescaled fields.
assert_zero((c - 10) / 32 * (c / 24)**2 / (c / 24)**3
            - sp.Rational(3, 4) * (1 - 10 / c))
g = sp.symbols("g", positive=True)
assert_zero((50 - 24 * (g + 1 / g) - c).subs(
    c, 50 - 24 * (g + 1 / g)))
root_vectors = sp.Matrix([[sp.Rational(1, 2), -sp.sqrt(3) / 2], [-1, 0]])
assert 2 * root_vectors * root_vectors.T == sp.Matrix([[2, -1], [-1, 2]])

# BHK identifies its Baxter variable t=lambda^3.  The spectral translation
# t=Lambda+z turns multiplicative shifts into finite additive shifts.  This
# checks arguments only; convergence/normalization of Q and T is unknown.
epsilon, z, power = sp.symbols("epsilon z power", real=True)
eta = sp.sqrt(3) * hbar / 16
small_g = eta * epsilon / sp.pi
shift_series = sp.series((1 / epsilon + z)
                         * sp.exp(sp.I * sp.pi * power * small_g)
                         - 1 / epsilon, epsilon, 0, 1).removeO()
assert_zero(shift_series - z - sp.I * power * eta)
soliton_k, soliton_v = sp.symbols("soliton_k soliton_v", real=True)
ell = soliton_v * (4 * soliton_k**2 - 1)
for sign in [-1, 1]:
    mu = (-soliton_v / sp.sqrt(3) + sign * soliton_k) / 2
    spectral_residual = sp.expand(mu**3 - mu / 4 + ell / (12 * sp.sqrt(3)))
    assert_zero(sp.rem(spectral_residual,
                       soliton_v**2 + soliton_k**2 - 1, soliton_v))
assert_zero(-12 * sp.sqrt(3) * (2 * sp.I * eta)
            + sp.Rational(9, 2) * sp.I * hbar)

numerics = []
for lambda_value in [10, 100, 1000]:
    numerical_c = -128 * mp.sqrt(3) * mp.pi * lambda_value
    substitution = {Lambda: lambda_value, hbar: 1, n: 1,
                    U0: sp.Rational(1, 4), c: chosen_c}
    gram = sp.Matrix([[UU_gram, cross_central], [cross_central, VV_gram]])
    gram = gram.subs(c, chosen_c).subs(substitution)
    values = mp.eigsy(mp.matrix([[mp.mpf(str(sp.N(gram[i, j], 82)))
                                 for j in range(2)] for i in range(2)]), eigvals_only=True)
    assert values[0] < 0 < values[1]
    exact_cross = mp.mpf(str(sp.N(canonical_cross.subs({hbar: 1, n: 1}), 82)))
    actual_cross = mp.mpf(str(sp.N(gram[0, 1], 82)))
    assert abs(actual_cross - exact_cross) < mp.mpf("1e-75")
    numerical_g = 48 / (50 - numerical_c
                       + mp.sqrt((2 - numerical_c) * (98 - numerical_c)))
    assert abs(50 - 24 * (numerical_g + 1 / numerical_g) - numerical_c) \
        < mp.mpf("1e-70")
    numerics.append({
        "Lambda": lambda_value,
        "central_charge": mp.nstr(numerical_c, 60),
        "g": mp.nstr(numerical_g, 60),
        "gamma": mp.nstr(mp.pi * numerical_g, 60),
        "gram_eigenvalues": [mp.nstr(value, 60) for value in values],
    })

large_lambda = mp.mpf("1e20")
large_c = -128 * mp.sqrt(3) * mp.pi * large_lambda
small_quantum_g = 48 / (50 - large_c
                        + mp.sqrt((2 - large_c) * (98 - large_c)))
shift_errors = []
for exponent in [mp.mpf(3), mp.mpf(1), mp.mpf("0.5"),
                 mp.mpf("-0.5"), mp.mpf(-1), mp.mpf(-3)]:
    spectral_z = mp.mpf("0.7")
    shifted_z = (large_lambda + spectral_z) * mp.exp(
        1j * mp.pi * exponent * small_quantum_g) - large_lambda
    expected_z = spectral_z + 1j * exponent * mp.sqrt(3) / 16
    error = abs(shifted_z - expected_z)
    assert error < mp.mpf("3e-20")
    shift_errors.append(mp.nstr(error, 40))

result = {
    "scope": "Formal Poisson and mode-algebra contraction, not a physical Hilbert-space construction",
    "checks": [
        "Second GD bracket independently derived from Miura Frechet operator",
        "Formal skew-adjointness of every differential-operator block",
        "Real good PDE and canonical-energy dictionaries",
        "Large-V Poisson-pencil contraction to canonical first bracket",
        "Hamiltonian generates the target flow under contracted bracket",
        "Fixed-vacuum determinant and Fourier inertia obstructions",
        "Real constant Miura fields obey the discriminant bound violated by large V",
        "Exact W3 mode central term gives physical canonical hbar scaling",
        "Highest-weight Gram matrix becomes indefinite",
        "Quantum derivative coefficient and A2 root-field normalization",
        "Multiplicative Baxter arguments have finite additive spectral-shift limits",
        "Common vacuum cubic root z=-ell/(12*sqrt(3)) gives Baxter step Delta ell=-9i*hbar/2",
        "80-digit finite-parameter Gram and g-normalization checks",
    ],
    "formulas": {
        "U": "1/4-3*u/2", "V": "sqrt(3)*pi/4",
        "tau": "sqrt(3)*t", "kappa": "sqrt(3)/(8*Lambda)",
        "c": "-128*sqrt(3)*pi*Lambda/hbar", "gamma": "pi*g",
        "g_relation": "c=50-24*(g+1/g)",
        "determinant_ratio": str(determinant_ratio),
        "constant_Miura_discriminant": "4*U^3-27*V^2=4*a^2*(a^2-3*b^2)^2>=0",
        "VV_highest_weight_commutator": str(VV_gram),
        "physical_density": "(8/27)*(U^3+9*V^2+3*U_x^2/4-3*U/16+1/32)",
        "Baxter_argument_limit": "t=Lambda+z, t*q^a-Lambda -> z+i*a*sqrt(3)*hbar/16",
        "soliton_spectral_dictionary": "z=-ell/(12*sqrt(3))",
        "Baxter_ell_step": "-9*i*hbar/2",
    },
    "failures_recorded": [
        "Direct same-background identification fails because UU brackets disagree",
        "Real invertible translation-invariant same-vacuum linear map fails Fourier inertia",
        "Inherited highest-weight vacuum fails positive-norm Gram test after large-V contraction",
        "Large V at fixed U leaves the real constant Miura-background domain",
    ],
    "implementation_failures": [],
    "physical_S_matrix": None,
    "runtime_seconds": time.monotonic() - START,
    "numerics": numerics,
    "Baxter_argument_errors_at_Lambda_1e20": shift_errors,
}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print("Miura derivation, canonical contraction and representation obstruction verified")
print("ALL CHECKS PASSED")
