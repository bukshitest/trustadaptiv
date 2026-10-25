import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 11, 'axes.linewidth': 0.9})
C1, C2, CR = '#2b6cb0', '#9fb3c8', '#8b2a2a'
R = json.load(open('results_A.json'))
c = R['conv']


def gm_trace(traces):
    L = int(np.median([len(t) for t in traces]))
    out = []
    for k in range(L):
        v = [t[k] for t in traces if len(t) > k]
        out.append(np.exp(np.mean(np.log(v))))
    return np.array(out)


# Fig. 5
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=250)
ta, ts = gm_trace(c['trace_adaptive']), gm_trace(c['trace_static'])
ax.semilogy(range(len(ta)), ta, '-o', color=C1, ms=4, lw=1.8, label='Adaptive weights (Eq. 3)')
ax.semilogy(range(len(ts)), ts, '-s', color=C2, ms=4, lw=1.8, label='Static weights')
ax.axhline(0.01, ls='--', color=CR, lw=1.2)
ax.text(0.4, 0.0115, 'tolerance ε = 0.01', color=CR, fontsize=10)
ax.set_xlabel('Consensus iteration'); ax.set_ylabel('Max pairwise divergence Δ(t)')
ax.grid(True, which='both', alpha=0.3); ax.legend(frameon=True)
fig.tight_layout(); fig.savefig('fig5.png'); plt.close(fig)

# Fig. 6
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.75), dpi=250)
for ax, key, lab, tag in ((axs[0], 'alpha', 'Consensus coefficient α', '(a)'),
                          (axs[1], 'eta', 'Trust adaptation rate η', '(b)')):
    xs = sorted(R['sens'][key], key=float)
    m = [np.mean(R['sens'][key][x]) for x in xs]
    s = [np.std(R['sens'][key][x], ddof=1) for x in xs]
    ax.errorbar([float(x) for x in xs], m, yerr=s, fmt='-o', color=C1, capsize=4, lw=1.8, ms=6)
    ax.set_xlabel(f'{lab}\n{tag}'); ax.set_ylabel('Iterations to reach ε')
    ax.set_ylim(0, max(m) * 1.25); ax.grid(True, alpha=0.3)
fig.tight_layout(w_pad=3); fig.savefig('fig6.png'); plt.close(fig)
print('ok', len(ta), len(ts))
