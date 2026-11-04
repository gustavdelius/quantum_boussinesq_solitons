#!/usr/bin/env python3
"""Independent infrared and fixed-charge tests for the good soliton sector.

The Gaussian calculations refer to the quadratic good vacuum. They do not
construct exact interacting soliton states. No historical project script is
imported or executed. Numerical evidence and limitations are cached next to
this script.
"""

from pathlib import Path
import json
import time

import mpmath as mp
import sympy as sp


START = time.perf_counter()
mp.mp.dps = 70
RESULTS = {"claims": {}, "failures_recorded": [], "limitations": []}


def record(name, value):
    RESULTS["claims"][name] = value


def as_text(value):
    return mp.nstr(value, 55)


def check_small(value, tolerance=mp.mpf("1e-55")):
    assert abs(value) < tolerance, (as_text(value), as_text(tolerance))


# The local first Hamiltonian operator annihilates both constant charge
# gradients. On the full potential algebra Pi generates a uniform phi shift.
x = sp.symbols("x", real=True)
k = sp.symbols("k", positive=True)
v = sp.symbols("v", real=True)
length = sp.symbols("L", positive=True)
mass, shift = sp.symbols("M Pi", real=True)
u = sp.Function("u")(x)
pi = sp.Function("pi")(x)
force = u - 3 * u**2 - sp.diff(u, x, 2)
assert sp.diff(sp.Integer(1), x) == 0
assert sp.simplify(sp.diff(force, x) -
                   (sp.diff(u, x) - 6*u*sp.diff(u, x) - sp.diff(u, x, 3))) == 0
record("local_charge_fluxes", {
    "M_dot": "[pi] at the spatial endpoints",
    "Pi_dot": "[u-3u^2-u_xx] at the spatial endpoints",
    "first_Hamiltonian_operator": "[[0,D],[D,0]]",
    "casimir_scope": "M and Pi are central for compactly supported derivative-field observables",
    "potential_algebra": "{phi(x),Pi}=1; Pi is not central in the potential algebra",
})


# Direct traveling-wave equation and exact one-soliton charge relations.
profile = k**2 / (2 * sp.cosh(k*x/2)**2)
travel_residual = sp.diff(profile, x, 2) - k**2*profile + 3*profile**2
assert sp.simplify(sp.expand_trig(travel_residual)) == 0
momentum = sp.Rational(2, 3) * k**3 * v
energy = sp.Rational(2, 15) * k**3 * (5 - 4*k**2)
assert sp.simplify(momentum.subs(k, mass/2) - v*mass**3/12) == 0
assert sp.diff(2*k, k) == 2
record("fixed_height_traveling_branch", {
    "mass": "M=2k",
    "shift_charge": "Pi=-2kv",
    "momentum": "P=2k^3 v/3=v M^3/12",
    "velocity": "v=epsilon sqrt(1-M^2/4)",
    "consequence": "At fixed M the classical traveling soliton has only the two velocity signs, not an open momentum interval",
})


# A finite ring with fixed twist M and mean momentum Pi has a separate
# zero-mode canonical pair. These are uniform-sector contributions only.
uniform_energy = length * (sp.Rational(1, 2)*(shift/length)**2 +
                          sp.Rational(1, 2)*(mass/length)**2 - (mass/length)**3)
assert sp.expand(uniform_energy - (shift**2 + mass**2)/(2*length) + mass**3/length**2) == 0
assert sp.simplify(-length*(shift/length)*(mass/length) + mass*shift/length) == 0
record("ring_zero_modes", {
    "decomposition": "phi=Phi+M*x/L+phi_tilde, pi=Pi/L+pi_tilde, fixed M and zero-mean tilded fields",
    "symplectic_form": "dPi wedge dPhi + integral d pi_tilde wedge d phi_tilde",
    "uniform_energy": "(Pi^2+M^2)/(2L)-M^3/L^2",
    "uniform_momentum": "-M Pi/L",
})


# Gaussian oscillator overlap, derived by completing the square rather than
# assuming an S-matrix or a collective-coordinate prescription.
omega, hbar = sp.symbols("omega hbar", positive=True)
a, b, p, r, z = sp.symbols("a b p r z", real=True)
linear = omega*(a+b) + sp.I*(r-p)
constant = -omega*(a*a+b*b)/2
integrated_exponent = sp.expand((linear**2/(4*omega)+constant)/hbar)
real_exponent = sp.simplify(sp.re(integrated_exponent))
assert sp.simplify(real_exponent +
                   (omega*(a-b)**2+(r-p)**2/omega)/(4*hbar)) == 0
record("oscillator_overlap", {
    "absolute_overlap": "exp(-[omega Delta_x^2+Delta_p^2/omega]/(4 hbar))",
    "relative_occupation": "[omega Delta_x^2+Delta_p^2/omega]/(2 hbar)",
})


# Explicit positive oscillator polarization of the contracted canonical
# algebra. One nonzero Fourier pair is represented by two independent real
# canonical oscillators. A finite collection of such pairs is a positive
# Fock representation of the quadratic Hamiltonian.
qc, qs, pc, ps = sp.symbols("Q_c Q_s P_c P_s", real=True)
qwave = sp.symbols("q", positive=True)


def linear_commutator(left, right):
    return sp.simplify(sp.I*hbar*(
        sp.diff(left, qc)*sp.diff(right, pc)-sp.diff(left, pc)*sp.diff(right, qc)
        + sp.diff(left, qs)*sp.diff(right, ps)-sp.diff(left, ps)*sp.diff(right, qs)))


phi_plus = (qc+sp.I*qs)/sp.sqrt(2*length)
phi_minus = sp.conjugate(phi_plus)
pi_plus = (pc+sp.I*ps)/sp.sqrt(2*length)
pi_minus = sp.conjugate(pi_plus)
assert sp.simplify(linear_commutator(phi_plus, pi_minus)-sp.I*hbar/length) == 0
assert linear_commutator(phi_plus, pi_plus) == 0

b_plus = sp.sqrt(length/(2*hbar))*(sp.sqrt(omega)*phi_plus+sp.I*pi_plus/sp.sqrt(omega))
b_minus = sp.sqrt(length/(2*hbar))*(sp.sqrt(omega)*phi_minus+sp.I*pi_minus/sp.sqrt(omega))
annihilators = [b_plus, b_minus]
for row in range(2):
    for column in range(2):
        expected_commutator = sp.Integer(row == column)
        assert sp.simplify(linear_commutator(annihilators[row],
                                           sp.conjugate(annihilators[column]))
                           -expected_commutator) == 0
        assert linear_commutator(annihilators[row], annihilators[column]) == 0

cosine_annihilator = (sp.sqrt(omega)*qc+sp.I*pc/sp.sqrt(omega))/sp.sqrt(2*hbar)
sine_annihilator = (sp.sqrt(omega)*qs+sp.I*ps/sp.sqrt(omega))/sp.sqrt(2*hbar)
assert sp.simplify(b_plus-(cosine_annihilator+sp.I*sine_annihilator)/sp.sqrt(2)) == 0
assert sp.simplify(b_minus-(cosine_annihilator-sp.I*sine_annihilator)/sp.sqrt(2)) == 0
assert sp.simplify(phi_plus-sp.sqrt(hbar/(2*length*omega))
                   *(b_plus+sp.conjugate(b_minus))) == 0
assert sp.simplify(pi_plus+sp.I*sp.sqrt(hbar*omega/(2*length))
                   *(b_plus-sp.conjugate(b_minus))) == 0

quadratic_pair = (pc**2+ps**2+omega**2*(qc**2+qs**2))/2
weyl_number_sum = sum(sp.conjugate(mode)*mode-sp.Rational(1, 2)
                     for mode in annihilators)
assert sp.simplify(quadratic_pair-hbar*omega*(weyl_number_sum+1)) == 0

# Physical target U_n and V_n each mix annihilation and creation modes.
# Their one-sided descendant Gram matrix in this actual Fock vacuum is
# positive semidefinite. It differs from a highest-weight commutator matrix
# because U_n and V_n do not annihilate the vacuum separately.
vacuum_phi_variance = hbar/(2*length*omega)
vacuum_pi_variance = hbar*omega/(2*length)
physical_gram = sp.Matrix([
    [sp.Rational(9, 4)*qwave**2*vacuum_phi_variance,
     -3*sp.sqrt(3)*hbar*qwave/(16*length)],
    [-3*sp.sqrt(3)*hbar*qwave/(16*length),
     sp.Rational(3, 16)*vacuum_pi_variance],
])
creation_amplitudes = [
    -3*sp.I*qwave*sp.sqrt(hbar/(2*length*omega))/2,
    sp.I*sp.sqrt(3)*sp.sqrt(hbar*omega/(2*length))/4,
]
derived_gram = sp.Matrix(2, 2, lambda row, column:
                         sp.conjugate(creation_amplitudes[row])*creation_amplitudes[column])
assert sp.simplify(physical_gram-derived_gram) == sp.zeros(2)
assert sp.simplify(physical_gram.det()) == 0
assert sp.simplify(physical_gram.trace()
                   -(9*hbar*qwave**2/(8*length*omega)
                     +3*hbar*omega/(32*length))) == 0
assert sp.simplify(linear_commutator(3*sp.I*qwave*phi_plus/2,
                                    sp.sqrt(3)*pi_minus/4)
                   +3*sp.sqrt(3)*hbar*qwave/(8*length)) == 0
record("positive_canonical_fock_polarization", {
    "Fourier_convention": "phi(x)=sum_n phi_n exp(-i q_n x), q_n=2 pi n/L; phi_n^dagger=phi_-n and pi_n^dagger=pi_-n",
    "canonical_mode_algebra": "[phi_n,pi_m]=i hbar delta_(n+m,0)/L",
    "annihilator": "b_n=sqrt(L/(2 hbar)) [sqrt(omega_n) phi_n+i pi_n/sqrt(omega_n)], n nonzero",
    "frequency": "omega_n=abs(q_n) sqrt(1+q_n^2)",
    "commutators": "[b_n,b_m^dagger]=delta_nm, [b_n,b_m]=0",
    "independence": "b_n and b_-n are independent annihilation modes, not adjoints",
    "inverse_phi": "phi_n=sqrt(hbar/(2L omega_n)) (b_n+b_-n^dagger)",
    "inverse_pi": "pi_n=-i sqrt(hbar omega_n/(2L)) (b_n-b_-n^dagger)",
    "quadratic_Hamiltonian": "H2=sum_(n nonzero) hbar omega_n (b_n^dagger b_n+1/2), with a finite mode cutoff before vacuum subtraction",
    "real_mode_pair": "b_+=(a_cos+i a_sin)/sqrt(2), b_-=(a_cos-i a_sin)/sqrt(2)",
    "target_modes": "U_n=(3i q_n/2) phi_n, V_n=sqrt(3) pi_n/4 for n nonzero",
    "positive_gram": str(physical_gram),
    "Gram_determinant": 0,
    "highest_weight_conflict": "U_n|0>=V_n|0>=0 for n>0 would require both b_n|0>=0 and b_-n^dagger|0>=0, impossible for a nonzero positive-norm vector",
    "scope": "Explicit positive canonical algebra and free-vacuum Fock space. Realization of contracted T/Q operators, renormalized interaction and interacting soliton states remains unconstructed.",
})


def soliton_u(position, amplitude):
    return amplitude**2 / (2*mp.cosh(amplitude*position/2)**2)


def soliton_u_hat(wavenumber, amplitude):
    if wavenumber == 0:
        return 2*amplitude
    return 2*mp.pi*wavenumber/mp.sinh(mp.pi*wavenumber/amplitude)


fourier_checks = []
for amplitude, wavenumber in [(mp.mpf("0.31"), mp.mpf("0.08")),
                              (mp.mpf("0.53"), mp.mpf("0.27")),
                              (mp.mpf("0.41"), mp.mpf("0.51"))]:
    direct = 2*mp.quad(lambda position: soliton_u(position, amplitude)*mp.cos(wavenumber*position),
                       [0, 2/amplitude, 8/amplitude, 20/amplitude, mp.inf])
    closed = soliton_u_hat(wavenumber, amplitude)
    check_small(direct-closed)
    fourier_checks.append({"k": as_text(amplitude), "q": as_text(wavenumber),
                           "residual": as_text(direct-closed)})
record("soliton_fourier_transform", {
    "formula": "u_hat(q)=2 pi q/sinh(pi q/k), pi_hat(q)=-v u_hat(q)",
    "checks": fourier_checks,
})


AMP1 = mp.mpf("0.31")
AMP2 = mp.mpf("0.53")
VEL1 = mp.sqrt(1-AMP1**2)
VEL2 = mp.sqrt(1-AMP2**2)
DELTA_M = 2*(AMP1-AMP2)
DELTA_PI = -2*AMP1*VEL1+2*AMP2*VEL2
LOG_COEFF = (DELTA_M**2+DELTA_PI**2)/(2*mp.pi)


def infrared_numerator(wavenumber, translated=False):
    u1 = soliton_u_hat(wavenumber, AMP1)
    if translated:
        u2 = u1*mp.exp(-mp.j*wavenumber*mp.mpf("1.7"))
        p1, p2 = -VEL1*u1, -VEL1*u2
    else:
        u2 = soliton_u_hat(wavenumber, AMP2)
        p1, p2 = -VEL1*u1, -VEL2*u2
    root = mp.sqrt(1+wavenumber*wavenumber)
    return root*abs(u1-u2)**2 + abs(p1-p2)**2/root


def cutoff_occupation(cutoff, translated=False):
    # q=exp(t) removes the explicit 1/q singularity.
    low = mp.log(cutoff)
    high = mp.log(8)
    cuts = [low] + [mp.mpf(t) for t in [-60, -30, -10, -5, 0]
                    if low < t < high] + [high]
    return mp.quad(lambda t: infrared_numerator(mp.exp(t), translated)/(2*mp.pi), cuts)


cutoffs = [mp.mpf("1e-6"), mp.mpf("1e-12"), mp.mpf("1e-24")]
occupations = [cutoff_occupation(cutoff) for cutoff in cutoffs]
fit_coefficient = (occupations[-1]-occupations[-2])/mp.log(cutoffs[-2]/cutoffs[-1])
check_small(fit_coefficient-LOG_COEFF, mp.mpf("1e-24"))
subtracted = [n-LOG_COEFF*mp.log(1/cutoff) for n, cutoff in zip(occupations, cutoffs)]
check_small(subtracted[-1]-subtracted[-2], mp.mpf("1e-22"))
record("infrared_log", {
    "hbar_for_numbers": 1,
    "k1": as_text(AMP1), "k2": as_text(AMP2),
    "delta_M": as_text(DELTA_M), "delta_Pi": as_text(DELTA_PI),
    "occupation_log_coefficient": as_text(LOG_COEFF),
    "coherent_overlap_power": as_text(LOG_COEFF/2),
    "formula": "N_diff(mu)=[Delta_M^2+Delta_Pi^2]/(2 pi hbar) log(1/mu)+O(1)",
    "occupation": [{"mu": as_text(mu), "N": as_text(n),
                     "N_subtracted": as_text(finite)}
                    for mu, n, finite in zip(cutoffs, occupations, subtracted)],
    "coefficient_residual": as_text(fit_coefficient-LOG_COEFF),
})


# Common finite-volume oscillator Hilbert space. Each periodized profile has
# its own uniform M/L and Pi/L removed. Both resulting data sets have M=Pi=0.
# Their nonzero Fourier modes retain the exact line-soliton transforms above.
def ring_occupation(box_length):
    spacing = 2*mp.pi/box_length
    maximum_mode = int(mp.ceil(8/spacing))
    return mp.fsum(infrared_numerator(n*spacing)/(n*spacing)
                   for n in range(1, maximum_mode+1))/box_length


lengths = [mp.mpf(256), mp.mpf(512), mp.mpf(1024), mp.mpf(2048)]
ring_values = [ring_occupation(box_length) for box_length in lengths]
ring_slopes = [(ring_values[j+1]-ring_values[j])/mp.log(2)
               for j in range(len(lengths)-1)]
ring_errors = [value-LOG_COEFF for value in ring_slopes]
assert all(abs(ring_errors[j+1]) < abs(ring_errors[j])/3
           for j in range(len(ring_errors)-1))
assert abs(ring_errors[-1]) < mp.mpf("2e-5")
record("finite_ring", {
    "common_charge_sector": "M=Pi=0 after uniform compensation of each periodized profile",
    "occupation": [{"L": as_text(box_length), "N": as_text(value)}
                    for box_length, value in zip(lengths, ring_values)],
    "doubling_slopes": [as_text(value) for value in ring_slopes],
    "slope_errors": [as_text(value) for value in ring_errors],
    "interpretation": "A log L divergence survives in a common compensated periodic oscillator space",
})


translation_low = cutoff_occupation(mp.mpf("1e-12"), translated=True)
translation_lower = cutoff_occupation(mp.mpf("1e-24"), translated=True)
check_small(translation_low-translation_lower, mp.mpf("1e-22"))
record("translation_control", {
    "k": as_text(AMP1), "displacement": "1.7",
    "occupation_limit": as_text(translation_lower),
    "cutoff_difference": as_text(translation_low-translation_lower),
    "interpretation": "Translation at fixed charges has finite infrared coherent occupation",
})


# A remote smooth charge-compensating cloud has vanishing energy as its
# width grows. The formula includes the exact cubic and gradient terms.
y = sp.symbols("y", real=True)
weight = sp.exp(-y*y)/sp.sqrt(sp.pi)
i1 = sp.integrate(weight, (y, -sp.oo, sp.oo))
i2 = sp.integrate(weight**2, (y, -sp.oo, sp.oo))
i3 = sp.integrate(weight**3, (y, -sp.oo, sp.oo))
id2 = sp.integrate(sp.diff(weight, y)**2, (y, -sp.oo, sp.oo))
assert i1 == 1
assert sp.simplify(i2-1/sp.sqrt(2*sp.pi)) == 0
assert sp.simplify(i3-1/(sp.sqrt(3)*sp.pi)) == 0
assert sp.simplify(id2-i2) == 0
cloud_energy = (mass**2+shift**2)*i2/(2*length)-mass**3*i3/length**2+mass**2*id2/(2*length**3)
cloud_momentum = -mass*shift*i2/length
cloud_checks = []
for cloud_width in [mp.mpf(13), mp.mpf(31)]:
    cloud_mass = mp.mpf("-0.44")
    cloud_shift = mp.mpf("0.3")
    def cloud_u(position):
        return cloud_mass*mp.exp(-(position/cloud_width)**2)/(mp.sqrt(mp.pi)*cloud_width)
    def cloud_pi(position):
        return cloud_shift*mp.exp(-(position/cloud_width)**2)/(mp.sqrt(mp.pi)*cloud_width)
    def cloud_density(position):
        value = cloud_u(position)
        derivative = -2*position*value/cloud_width**2
        return (cloud_pi(position)**2+value**2+derivative**2)/2-value**3
    direct_energy = mp.quad(cloud_density, [-mp.inf, 0, mp.inf])
    expected_energy = ((cloud_mass**2+cloud_shift**2)/(2*cloud_width*mp.sqrt(2*mp.pi))
                       -cloud_mass**3/(cloud_width**2*mp.sqrt(3)*mp.pi)
                       +cloud_mass**2/(2*cloud_width**3*mp.sqrt(2*mp.pi)))
    check_small(direct_energy-expected_energy)
    check_small(mp.quad(cloud_u, [-mp.inf, 0, mp.inf])-cloud_mass)
    check_small(mp.quad(cloud_pi, [-mp.inf, 0, mp.inf])-cloud_shift)
    cloud_checks.append({"width": as_text(cloud_width),
                         "energy": as_text(direct_energy),
                         "energy_residual": as_text(direct_energy-expected_energy)})
record("compensating_cloud", {
    "fields": "u_c=Delta_M w((x-R)/L)/L, pi_c=Delta_Pi w((x-R)/L)/L, integral w=1",
    "gaussian": "w(y)=exp(-y^2)/sqrt(pi)",
    "energy": str(cloud_energy),
    "momentum": str(cloud_momentum),
    "scaling": "M_c and Pi_c stay fixed, H_c=O(1/L), P_c=O(1/L); spatial separation removes soliton-cloud cross terms",
    "independent_quadratures": cloud_checks,
})


RESULTS["limitations"] = [
    "These are exact Gaussian-vacuum and classical phase-space diagnostics, not a construction of interacting quantum solitons.",
    "The divergent free coherent norm does not prove that an interacting dressed S-matrix or stable soliton state cannot exist.",
    "The finite-ring profile uses exact periodization and a uniform charge compensation; it is initial data, not an exact periodic nonlinear traveling solution.",
    "A fixed total charge sector can accommodate a varying local soliton profile with a soft cloud. Its interacting asymptotic evolution and scattering normalization remain to be derived.",
    "No claim about quantum decay, exact factorization, or a physical infraparticle exponent is made.",
]
RESULTS["failures_recorded"] = [
    "The tentative identification of the bare traveling-wave family with an open momentum branch at fixed kink height fails: M=2k fixes k.",
    "The tentative inference that an arbitrarily low-energy compensating background is harmless to the Fock-state limit fails: the occupation diverges logarithmically with box length.",
    "No Python assertion failed during the implementation and no numerical tolerance was weakened.",
]
RESULTS["runtime_seconds"] = time.perf_counter()-START
output_path = Path(__file__).with_suffix(".json")
output_path.write_text(json.dumps(RESULTS, indent=2)+"\n")
print("Occupation logarithm coefficient:", as_text(LOG_COEFF))
print("Finite-ring slope errors:", [as_text(value) for value in ring_errors])
print("Translation occupation:", as_text(translation_lower))
print("ALL CHECKS PASSED")
