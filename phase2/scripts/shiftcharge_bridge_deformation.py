#!/usr/bin/env python3
"""Independent current-deformation checks for the canonical good Boussinesq model.

The exact field transformation and all numerical tests here are classical.
The proposed quantum unitary and its scattering interpretation remain conditional.
No historical repository script is imported or executed.
"""

from pathlib import Path
import json
import time

import mpmath as mp
import sympy as sp


started = time.monotonic()
mp.mp.dps = 80
checks = []


def exact(name, expression):
    remainder = sp.simplify(sp.expand(expression))
    assert remainder == 0, (name, remainder)
    checks.append({"name": name, "kind": "symbolic", "remainder": "0"})


def near(name, actual, expected, tolerance):
    error = abs(actual - expected)
    assert error < tolerance, (name, mp.nstr(error, 40), tolerance)
    checks.append({"name": name, "kind": "mpmath", "error": mp.nstr(error, 20),
                   "tolerance": str(tolerance)})


# Local conservation laws, the generator, and all integration-by-parts terms.
x = sp.symbols("x", real=True)
chi = sp.Function("chi")(x)
pi = sp.Function("pi")(x)
u = sp.diff(chi, x)
ux = sp.diff(u, x)
uxx = sp.diff(u, x, 2)
elastic_force = u - 3*u**2 - uxx
pi_t = sp.diff(elastic_force, x)
u_t = sp.diff(pi, x)
momentum_density = -pi*u
mass_current = -pi
momentum_current = (pi**2 + u**2 + ux**2)/2 - 2*u**3 - u*uxx
hamiltonian_density = (pi**2 + u**2 + ux**2)/2 - u**3
exact("mass continuity", u_t + sp.diff(mass_current, x))
exact("momentum continuity", -pi_t*u-pi*u_t+sp.diff(momentum_current, x))

deformation_current = u*momentum_current-mass_current*momentum_density
deformation_bulk = -pi**2*u/2+u**3/2-2*u**4+sp.Rational(5, 2)*u*ux**2
exact("current density boundary term", deformation_bulk-deformation_current
      - sp.diff(u**2*ux, x))

generator_density = chi*pi*u
generator_phi_gradient = sp.diff(generator_density, chi)-sp.diff(
    sp.diff(generator_density, u), x)
generator_pi_gradient = sp.diff(generator_density, pi)
exact("generator phi gradient", generator_phi_gradient + chi*sp.diff(pi, x))
exact("generator pi gradient", generator_pi_gradient - chi*u)
phi_g = chi*u
pi_g = chi*sp.diff(pi, x)
u_g = sp.diff(phi_g, x)
exact("mass fixed by flow", u_g-sp.diff(chi*u, x))
exact("momentum fixed by flow", -pi_g*u-pi*u_g+sp.diff(chi*pi*u, x))
exact("shift charge flow boundary", pi_g-momentum_density-sp.diff(chi*pi, x))
variation_h = pi*pi_g+(u-3*u**2)*u_g+ux*sp.diff(u_g, x)
exact("Hamiltonian canonical flow", variation_h-deformation_bulk
      -sp.diff(chi*hamiltonian_density, x))
generator_t = pi*pi*u+chi*pi_t*u+chi*pi*u_t
exact("generator time derivative", generator_t+deformation_current
      -sp.diff(chi*momentum_current, x))

# Exact reciprocal-coordinate transformation and rational Hamiltonian family.
g, h, a, b, p = sp.symbols("g h a b p", real=True)
finite_h = p**2*(1-g*a)/2+a**2/(2*(1-g*a))-a**3/(1-g*a)**2+b**2/(2*(1-g*a)**5)
exact("finite Hamiltonian derivative", sp.diff(finite_h, g).subs(g, 0)
      -(-p**2*a/2+a**3/2-2*a**4+sp.Rational(5, 2)*a*b**2))
exact("reciprocal Jacobian composition", (a/(1-h*a))/(1-g*a/(1-h*a))
      -a/(1-(g+h)*a))
exact("gradient transformation", sp.diff(a/(1-g*a), a)*b/(1-g*a)
      -b/(1-g*a)**3)

# For a solution of H_g, use inverse T_g: x=y+g chi_0(y), pi=pi_0.
jacobian = 1+g*a
inverse_h = finite_h.subs({a: a/jacobian, b: b/jacobian**3}, simultaneous=True)
exact("energy of inverse transformed profile", jacobian*inverse_h
      -(p**2+a**2+b**2)/2+a**3)
exact("mass of inverse transformed profile", jacobian*a/jacobian-a)
exact("momentum of inverse transformed profile", -jacobian*p*a/jacobian+p*a)
exact("deformed velocity law", sp.diff(finite_h, p)-(1-g*a)*p)

# Symplectic two-form: at fixed new x, delta chi_g=delta chi_0/(1-gu),
# delta pi_g=delta pi_0 + g*pi'_0*delta chi_0/(1-gu).
# The second term wedges to zero, and the Jacobian cancels the first.
exact("symplectic wedge coefficient", (1-g*a)/(1-g*a)-1)
exact("vacuum elastic curvature unchanged", sp.diff(finite_h.subs({p: 0, b: 0}), a, 2)
      .subs(a, 0)-1)
exact("deformation has no quadratic vacuum term",
      sp.series(finite_h.subs({p: 0, b: 0}), a, 0, 3).removeO()-a**2/2)

k, v = sp.symbols("k v", positive=True, real=True)
int_u3 = 4*k**5/15
int_u4 = 4*k**7/35
exact("zero one-soliton energy perturbation", 3*k**2*int_u3-7*int_u4)

P1, P2, M1, M2, Ptotal, Mtotal, L, hbar, c = sp.symbols(
    "P1 P2 M1 M2 Ptotal Mtotal L hbar c", real=True)
wedge = P1*M2-P2*M1
xin = -(M1*P1+M2*P2)/2-M1*P2
xout = -(M1*P1+M2*P2)/2-M2*P1
exact("asymptotic generator difference", xout-xin+wedge)
exact("one-loop phase parameter", (-g*wedge/hbar).subs(g, -hbar*c)-c*wedge)
exact("antisymmetry of dressing phase", wedge+wedge.xreplace(
    {P1: P2, P2: P1, M1: M2, M2: M1}))
exact("ring length and twist algebra", P1*L-g*(P1*Mtotal-M1*Ptotal)
      -(P1*(L-g*Mtotal)+g*M1*Ptotal))


def soliton_values(position, kval):
    u0 = kval**2/(2*mp.cosh(kval*position/2)**2)
    ux0 = -kval*mp.tanh(kval*position/2)*u0
    phi0 = kval*(1+mp.tanh(kval*position/2))
    return phi0, u0, ux0


onebody = []
for kval, gval in [(mp.mpf("0.37"), mp.mpf("0.3")),
                   (mp.mpf("0.61"), mp.mpf("-0.2"))]:
    velocity = mp.sqrt(1-kval**2)
    mass = 2*kval
    momentum = mp.mpf(2)/3*velocity*kval**3
    energy = mp.mpf(2)/15*kval**3*(5-4*kval**2)
    shift_charge = -mass*velocity
    breaks = [-mp.inf, -8/kval, 0, 8/kval, mp.inf]

    def original_generator(position):
        phi0, u0, _ = soliton_values(position, kval)
        return -phi0*velocity*u0**2

    def deformation_on_profile(position):
        _, u0, ux0 = soliton_values(position, kval)
        return -velocity**2*u0**3/2+u0**3/2-2*u0**4+mp.mpf(5)/2*u0*ux0**2

    def physical_charges(position):
        phi0, u0, ux0 = soliton_values(position, kval)
        jac = 1+gval*u0
        physical_u = u0/jac
        physical_ux = ux0/jac**3
        physical_pi = -velocity*u0
        denom = 1-gval*physical_u
        density_h = (physical_pi**2*denom/2+physical_u**2/(2*denom)
                     -physical_u**3/denom**2+physical_ux**2/(2*denom**5))
        return (jac*physical_u, -jac*physical_pi*physical_u,
                jac*physical_pi, jac*density_h,
                jac*(position+gval*phi0)*physical_u)

    near("one-soliton X self term", mp.quad(original_generator, breaks),
         -mass*momentum/2, mp.mpf("1e-65"))
    near("one-soliton integrated deformation", mp.quad(deformation_on_profile, breaks),
         0, mp.mpf("1e-65"))
    values = [mp.quad(lambda z, i=i: physical_charges(z)[i], breaks) for i in range(5)]
    expected = [mass, momentum, shift_charge-gval*momentum, energy, gval*mass**2/2]
    for name, actual, reference in zip(["M", "P", "Pi", "E", "centroid numerator"], values, expected):
        near("inverse transformed soliton "+name, actual, reference, mp.mpf("1e-65"))
    near("deformed centroid Ward identity", values[2]+gval*values[1],
         -mass*velocity, mp.mpf("1e-65"))
    onebody.append({"k": str(kval), "g": str(gval),
                    "charges": [mp.nstr(z, 40) for z in values]})


# Independent full two-soliton tau calculation. Smooth tau has positive A.
k1, k2 = mp.mpf("0.25"), mp.mpf("0.60")
v1, v2 = mp.sqrt(1-k1**2), mp.sqrt(1-k2**2)
w1, w2 = k1*v1, k2*v2
bilinear = lambda kval, wval: wval**2-kval**2+kval**4
A = -bilinear(k1-k2, w1-w2)/bilinear(k1+k2, w1+w2)
assert 0 < A < 1
ks = [mp.mpf(0), k1, k2, k1+k2]
ts = [mp.mpf(0), -w1, -w2, -w1-w2]


def tau_fields(position, time_value):
    logs = [mp.mpf(0), k1*position-w1*time_value, k2*position-w2*time_value,
            mp.log(A)+(k1+k2)*position-(w1+w2)*time_value]
    largest = max(logs)
    weights = [mp.exp(z-largest) for z in logs]
    total = sum(weights)
    weights = [z/total for z in weights]
    mean_k = sum(weight*kval for weight, kval in zip(weights, ks))
    mean_t = sum(weight*tval for weight, tval in zip(weights, ts))
    dk = [z-mean_k for z in ks]
    dt = [z-mean_t for z in ts]
    moment = lambda nk, nt: sum(weight*dx**nk*dy**nt
                                for weight, dx, dy in zip(weights, dk, dt))
    phi = 2*mean_k
    u0 = 2*moment(2, 0)
    pi0 = 2*moment(1, 1)
    ux0 = 2*moment(3, 0)
    uxxx0 = 2*(moment(5, 0)-10*moment(3, 0)*moment(2, 0))
    ut0 = 2*moment(2, 1)
    pit0 = 2*moment(1, 2)
    return phi, u0, pi0, ux0, uxxx0, ut0, pit0


for position, time_value in [(mp.mpf("-3"), mp.mpf("0")),
                             (mp.mpf("0.4"), mp.mpf("1.3")),
                             (mp.mpf("5"), mp.mpf("-2"))]:
    _, u0, _, ux0, uxxx0, _, pit0 = tau_fields(position, time_value)
    near("two-soliton canonical field equation", pit0, ux0-6*u0*ux0-uxxx0,
         mp.mpf("1e-70"))


def integration_points(time_value):
    centers = [v1*time_value, v2*time_value,
               v1*time_value-mp.log(A)/k1, v2*time_value-mp.log(A)/k2]
    points = [mp.mpf(0)] if time_value == 0 else []
    for center in centers:
        points.extend(center+offset/k1 for offset in [-16, -8, 0, 8, 16])
    return [-mp.inf]+sorted(set(points))+[mp.inf]


def generator_integral(time_value):
    def density(position):
        phi, u0, pi0, *_ = tau_fields(position, time_value)
        return phi*pi0*u0
    return mp.quad(density, integration_points(time_value))


mass1, mass2 = 2*k1, 2*k2
momentum1, momentum2 = mp.mpf(2)/3*v1*k1**3, mp.mpf(2)/3*v2*k2**3
self_part = -(mass1*momentum1+mass2*momentum2)/2
xin_expected = self_part-mass1*momentum2
xout_expected = self_part-mass2*momentum1
time_far = mp.mpf("2500")
xin_value = generator_integral(-time_far)
xout_value = generator_integral(time_far)
near("two-soliton incoming bilocal factorization", xin_value, xin_expected,
     mp.mpf("1e-37"))
near("two-soliton outgoing bilocal factorization", xout_value, xout_expected,
     mp.mpf("1e-37"))
near("two-soliton action boundary wedge", xout_value-xin_value,
     -(momentum1*mass2-momentum2*mass1), mp.mpf("1e-37"))


def collision_current(position):
    _, u0, pi0, ux0, *_ = tau_fields(position, 0)
    return -pi0**2*u0/2+u0**3/2-2*u0**4+mp.mpf(5)/2*u0*ux0**2


def collision_generator_derivative(position):
    phi, u0, pi0, _, _, ut0, pit0 = tau_fields(position, 0)
    return pi0*pi0*u0+phi*pit0*u0+phi*pi0*ut0


collision_deformation = mp.quad(collision_current, integration_points(0))
collision_xdot = mp.quad(collision_generator_derivative, integration_points(0))
near("collision integrated current versus X derivative", collision_deformation,
     -collision_xdot, mp.mpf("1e-65"))


# A finite-interval test retains nonzero endpoint currents and a moving anchor.
left, right = mp.mpf("-1.2"), mp.mpf("0.8")
test_phi = lambda z: mp.mpf("0.3")*mp.tanh(z)+mp.mpf("0.1")*mp.sin(z)
test_pi = lambda z: mp.mpf("0.2")*mp.exp(-z*z)+mp.mpf("0.1")*mp.cos(z)


def finite_fields(z):
    field_u = mp.diff(test_phi, z)
    field_ux = mp.diff(test_phi, z, 2)
    field_uxx = mp.diff(test_phi, z, 3)
    field_pi = test_pi(z)
    field_pit = field_ux-6*field_u*field_ux-mp.diff(test_phi, z, 4)
    field_jp = (field_pi**2+field_u**2+field_ux**2)/2-2*field_u**3-field_u*field_uxx
    density_d = field_u*field_jp-field_pi**2*field_u
    anchored_phi = test_phi(z)-test_phi(left)
    density_xdot = ((field_pi-test_pi(left))*field_pi*field_u
                    +anchored_phi*field_pit*field_u
                    +anchored_phi*field_pi*mp.diff(test_pi, z))
    return field_u, field_pi, field_jp, density_d, density_xdot


finite_p = mp.quad(lambda z: -finite_fields(z)[0]*finite_fields(z)[1], [left, right])
finite_m = test_phi(right)-test_phi(left)
finite_d = mp.quad(lambda z: finite_fields(z)[3], [left, right])
finite_xdot = mp.quad(lambda z: finite_fields(z)[4], [left, right])
finite_boundary = -test_pi(left)*finite_p-finite_m*finite_fields(right)[2]
near("finite interval current boundary term", -finite_xdot, finite_d+finite_boundary,
     mp.mpf("1e-65"))

results = {
    "scope": "Classical exact deformation and conditional semiclassical dressing; no quantum domain constructed",
    "sources_read": [
        {"id": "arXiv:2105.03326v5", "reading": "Full-text sections 2-3, Appendix B, Appendix C relevant boundary formulas"},
        {"id": "arXiv:2010.15733v3", "reading": "Full-text sections 2.2, 3.2-3.3, 3.6"}],
    "checks": checks,
    "onebody": onebody,
    "two_soliton": {"k1": str(k1), "k2": str(k2), "time_far": str(time_far),
        "A": mp.nstr(A, 40), "X_in": mp.nstr(xin_value, 40),
        "X_out": mp.nstr(xout_value, 40), "X_difference": mp.nstr(xout_value-xin_value, 40),
        "D_at_collision": mp.nstr(collision_deformation, 40)},
    "finite_interval": {"left": str(left), "right": str(right),
        "boundary_term": mp.nstr(finite_boundary, 40),
        "integrated_deformation": mp.nstr(finite_d, 40)},
    "failures_recorded": [],
    "physical_quantum_S": None,
    "physical_c_good": None,
    "runtime_seconds": time.monotonic()-started,
}
Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2)+"\n")
print(f"Verified {len(checks)} checks in {results['runtime_seconds']:.2f} s")
print("ALL CHECKS PASSED")
