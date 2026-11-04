#!/usr/bin/env python3
"""Independent first-bracket charge and quantisation diagnostic.

The continuum conservation law is exact. The Weyl observation is a
finite-dimensional algebraic identity, not construction of a continuum
operator. A naive lattice counterexample is retained explicitly.
"""

import json
from pathlib import Path
import time

import mpmath as mp
import sympy as sp


def main():
    started = time.perf_counter()
    mp.mp.dps = 70
    u = sp.symbols("u0:10")
    p = sp.symbols("p0:10")
    sound_squared = sp.symbols("s", real=True)

    def derivative(expression):
        return sp.expand(sum(
            sp.diff(expression, u[index]) * u[index + 1]
            + sp.diff(expression, p[index]) * p[index + 1]
            for index in range(9)
        ))

    def derivatives(expression, order):
        for _ in range(order):
            expression = derivative(expression)
        return expression

    def euler(expression, jets):
        return sp.expand(sum(
            (-1) ** index * derivatives(sp.diff(expression, jets[index]), index)
            for index in range(5)
        ))

    coefficients = sp.symbols("a b c d e f g h")
    a, b, c, d, e, f, g, h = coefficients
    ansatz = (
        p[1] ** 2 + a * u[0] * p[0] ** 2 + b * u[0] ** 4
        + c * u[0] * u[1] ** 2 + d * u[2] ** 2
        + e * sound_squared * p[0] ** 2 + f * sound_squared * u[0] ** 3
        + g * sound_squared * u[1] ** 2 + h * sound_squared ** 2 * u[0] ** 2
    )
    velocity_u = p[1]
    velocity_p = sound_squared * u[1] - 6 * u[0] * u[1] - u[3]

    def time_derivative(density):
        return sp.expand(sum(
            sp.diff(density, u[index]) * derivatives(velocity_u, index)
            + sp.diff(density, p[index]) * derivatives(velocity_p, index)
            for index in range(3)
        ))

    ansatz_rate = time_derivative(ansatz)
    equations = (
        sp.Poly(euler(ansatz_rate, u), *u, *p, sound_squared).coeffs()
        + sp.Poly(euler(ansatz_rate, p), *u, *p, sound_squared).coeffs()
    )
    solutions = sp.solve(equations, coefficients, dict=True)
    assert len(solutions) == 1
    expected = {a: -3, b: 3, c: -9, d: 1, e: h, f: -2 * h - 1, g: h + 1}
    assert solutions[0] == expected
    density = sp.expand(ansatz.subs(solutions[0]).subs(h, 0))
    rate = time_derivative(density)
    assert euler(rate, u) == 0
    assert euler(rate, p) == 0
    assert sp.Poly(rate, *u, *p).coeff_monomial(1) == 0
    energy_density = (
        p[0] ** 2 / 2 + sound_squared * u[0] ** 2 / 2
        - u[0] ** 3 + u[1] ** 2 / 2
    )
    assert sp.expand(
        ansatz.subs(solutions[0]) - density - 2 * h * sound_squared * energy_density
    ) == 0

    # Independent exact soliton substitution and high-precision quadrature.
    k, v, y = sp.symbols("k v y", real=True)
    pulse = k ** 2 * (1 - y ** 2) / 2

    def pulse_derivative(expression):
        return sp.expand(k * (1 - y ** 2) * sp.diff(expression, y) / 2)

    substitution = {sound_squared: 1, u[0]: pulse, p[0]: -v * pulse}
    for index in range(1, 3):
        substitution[u[index]] = pulse_derivative(substitution[u[index - 1]])
        substitution[p[index]] = pulse_derivative(substitution[p[index - 1]])
    pulse_density = sp.expand(density.subs(substitution))
    pulse_integral = sp.factor(sp.integrate(
        sp.cancel(2 * pulse_density / (k * (1 - y ** 2))), (y, -1, 1)
    ))
    on_shell = sp.factor(pulse_integral.subs(v ** 2, 1 - k ** 2))
    assert on_shell == 4 * k ** 5 * (20 * k ** 2 - 21) / 105
    pulse_numeric = sp.lambdify((y, k, v),
                              sp.cancel(2 * pulse_density / (k * (1 - y ** 2))),
                              "mpmath")
    samples = []
    for sample_k in (mp.mpf("0.23"), mp.mpf("0.51"), mp.mpf("0.79")):
        sample_v = mp.sqrt(1 - sample_k ** 2)
        integral = mp.quad(lambda yy: pulse_numeric(yy, sample_k, sample_v), [-1, 0, 1])
        closed = 4 * sample_k ** 5 * (20 * sample_k ** 2 - 21) / 105
        error = abs(integral - closed)
        assert error < mp.mpf("1e-65")
        samples.append({"k": str(sample_k), "charge": str(integral), "error": str(error)})

    # A finite canonical regulator preserves the degree argument for the
    # third Moyal contraction, but need not preserve classical integrability.
    # This counterexample prevents the zero contraction being overinterpreted.
    size = 3
    phi = sp.symbols("phi0:3")
    momentum = sp.symbols("mom0:3")

    def lattice_derivative(values):
        return [(values[(index + 1) % size] - values[(index - 1) % size]) / 2
                for index in range(size)]

    strain = lattice_derivative(phi)
    strain_x = lattice_derivative(strain)
    strain_xx = lattice_derivative(strain_x)
    momentum_x = lattice_derivative(momentum)
    lattice_hamiltonian = sp.expand(sum(
        (momentum[index] ** 2 + strain[index] ** 2 + strain_x[index] ** 2) / 2
        - strain[index] ** 3 for index in range(size)
    ))
    lattice_charge = sp.expand(sum(
        momentum_x[index] ** 2 + strain_xx[index] ** 2
        - 3 * strain[index] * momentum[index] ** 2 + 3 * strain[index] ** 4
        - 9 * strain[index] * strain_x[index] ** 2
        + strain_x[index] ** 2 - strain[index] ** 3 for index in range(size)
    ))
    third_moyal = 0
    for first in range(size):
        for second in range(size):
            for third in range(size):
                h_derivative = sp.diff(lattice_hamiltonian, phi[first], phi[second], phi[third])
                q_derivative = sp.diff(lattice_charge, momentum[first], momentum[second], momentum[third])
                third_moyal += h_derivative * q_derivative
    assert sp.expand(third_moyal) == 0
    assert sp.Poly(lattice_hamiltonian, *phi, *momentum).total_degree() == 3
    # Every other third derivative of H involving momentum vanishes.
    for first in range(size):
        for second in range(2 * size):
            for third in range(2 * size):
                variables = phi + momentum
                assert sp.diff(lattice_hamiltonian, momentum[first],
                               variables[second], variables[third]) == 0
    lattice_bracket = sp.expand(sum(
        sp.diff(lattice_hamiltonian, phi[index]) * sp.diff(lattice_charge, momentum[index])
        - sp.diff(lattice_hamiltonian, momentum[index]) * sp.diff(lattice_charge, phi[index])
        for index in range(size)
    ))
    point = dict(zip(phi + momentum, (1, 2, 4, 2, -1, 3)))
    lattice_defect = sp.simplify(lattice_bracket.subs(point))
    assert lattice_defect != 0

    results = {
        "scope": "Exact continuum classical charge and finite-regulator Weyl-degree diagnostic",
        "density": str(density),
        "coefficient_solution": {str(key): str(value) for key, value in expected.items()},
        "continuum_euler_residuals": [0, 0],
        "soliton_charge": str(on_shell),
        "soliton_quadrature": samples,
        "third_moyal_contraction": 0,
        "naive_three_site_classical_bracket_at_point": str(lattice_defect),
        "failures_recorded": [
            "Naive periodic central-difference lattice does not preserve the extra charge; "
            "its nonzero bracket is an explicitly checked rejected regulator."
        ],
        "continuum_quantum_operator_constructed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print("Continuum charge conservation verified by independent Euler derivatives.")
    print(f"Rejected naive lattice bracket at test point: {lattice_defect}")
    print("Weyl degree check passed; renormalised continuum operator remains open.")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
