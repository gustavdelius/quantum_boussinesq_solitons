# (G) Pure figure/data generator; no verification claims are asserted here.
"""Figures for the Phase 1 review: interaction regimes in the rapidity-like variables and E(P)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g = np.linspace(1e-3, np.pi/2 - 1e-3, 600)
A1, A2 = np.meshgrid(g, g)
S, D = A1 + A2, A1 - A2
with np.errstate(divide='ignore', invalid='ignore'):
    Aov = (1 - 3/np.tan(S/2)**2)/(1 - 3/np.tan(D/2)**2)
    Aho = (1 - 3*np.tan(D/2)**2)/(1 - 3*np.tan(S/2)**2)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a_, Z, title in [(ax[0], Aov, 'good, overtaking ($\\epsilon_1=\\epsilon_2$)'),
                     (ax[1], Aho, 'good, head-on ($\\epsilon_1=-\\epsilon_2$)')]:
    cls = np.where(Z < 0, 0, np.where(Z < 1, 1, 2))
    a_.contourf(A1, A2, cls, levels=[-.5, .5, 1.5, 2.5], colors=['#d9534f', '#9ecae1', '#74c476'])
    a_.set_xlabel('$\\alpha_1$'); a_.set_ylabel('$\\alpha_2$'); a_.set_title(title, fontsize=10)
    a_.set_aspect('equal')
ax[0].text(0.25, 0.35, 'regular\n$0<A<1$', fontsize=9); ax[0].text(1.15, 1.3, 'singular', fontsize=9, color='w')
ax[1].text(0.03, 0.08, '$A>1$', fontsize=8); ax[1].text(0.8, 0.9, 'singular', fontsize=9, color='w')
ax[1].text(0.05, 1.3, 'regular\n$0<A<1$', fontsize=9)
k = np.linspace(0, 1, 400)
P = 2/3*k**3*np.sqrt(1 - k**2); E = 2/15*k**3*(5 - 4*k**2)
ax[2].plot(P[k <= np.sqrt(3)/2], E[k <= np.sqrt(3)/2], label='$k<\\sqrt{3}/2$')
ax[2].plot(P[k >= np.sqrt(3)/2], E[k >= np.sqrt(3)/2], '--', label='$k>\\sqrt{3}/2$')
ax[2].plot(P, P, ':', color='gray', label='$E=P$ (sound)')
ax[2].set_xlabel('$P$'); ax[2].set_ylabel('$E$'); ax[2].legend(fontsize=8)
ax[2].set_title('good: one-soliton $E(P)$, right-movers', fontsize=10)
fig.tight_layout(); fig.savefig('../figures/regimes_and_dispersion.pdf'); print('saved')

print("DATA GENERATED (no checks)")
