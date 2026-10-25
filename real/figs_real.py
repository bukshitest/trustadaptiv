import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 11, 'axes.linewidth': 0.9})
C1, C2, C3, C4, CR = '#2b6cb0', '#9fb3c8', '#6b8e23', '#c47f2c', '#8b2a2a'
o = json.load(open('v65_real.json')); e = json.load(open('v65_extra.json')); x7 = json.load(open('v7_extra.json')); bc = json.load(open('v8_bicheb.json'))
o['boot']['median'] = {'rs': x7['median_boot']['rs']}; o['orig']['median'] = x7['median_orig']
o['boot']['bic'] = {'rs': bc['boot']['rs']}; o['orig']['bic'] = bc['orig']
names = {'median': 'Coordinate-wise\nmedian', 'proposed': 'Proposed\n(trust-adaptive)', 'static': 'Static-weight\nconsensus', 'mean': 'Simple\naveraging', 'bi': 'Centralized BI\n(weighted sum)', 'bic': 'Centralized BI\n(rule (6))'}
M = ['proposed', 'static', 'mean', 'median', 'bi', 'bic']
# Fig 2
fig, ax = plt.subplots(figsize=(9.0, 3.7), dpi=250)
data = [o['boot'][m]['rs'] for m in M]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color='k'))
for p, c in zip(bp['boxes'], [C1, C2, C2, C3, C4, '#b5651d']): p.set_facecolor(c); p.set_alpha(0.75)
ax.scatter(range(1, 7), [o['orig'][m]['rs'] for m in M], marker='D', color=CR, zorder=3, label='Original sample')
ax.set_xticks(range(1, 7)); ax.set_xticklabels([names[m] for m in M]); ax.set_ylabel('Spearman $r_S$ with 2026 outcome')
ax.grid(True, axis='y', alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), frameon=False)
fig.tight_layout(); fig.savefig('fig2.png'); plt.close(fig)
# Fig 3
tr = o['trace']; D = [t[0] for t in tr]; W = [np.array(t[1]) for t in tr]
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=250)
ax = axs[0]; t = np.arange(len(D))
ax.semilogy(t, D, '-o', color=C1, ms=5, lw=1.8, label='$\\Delta^{(t)}$ (observed)')
th = 1 - 3 * 0.3 * 0.05; ax.semilogy(t, D[0] * th ** t, '--', color=C2, lw=1.5, label='bound $\\bar\\theta^{\\,t}\\Delta^{(0)}$')
ax.axhline(0.01, ls=':', color=CR, lw=1.2); ax.text(0.2, 0.0115, 'ε = 0.01', color=CR, fontsize=9)
ax.set_xlabel('Consensus iteration t\n(a)'); ax.set_ylabel('Maximal divergence'); ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)
ax = axs[1]
pairs = [((1, 0), 'ANA-2 → ANA-1'), ((2, 0), 'ANA-3 → ANA-1'), ((1, 2), 'ANA-2 → ANA-3'), ((2, 1), 'ANA-3 → ANA-2')]
for (i, k), lab in pairs:
    ax.plot(t, [w[i, k] for w in W], '-o', ms=3.5, lw=1.5, label=lab)
ax.set_xlabel('Consensus iteration t\n(b)'); ax.set_ylabel('Trust weight $w_{i,i\\prime}$'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8.5)
fig.tight_layout(w_pad=3); fig.savefig('fig3.png'); plt.close(fig)
# Fig 4  (ANA-1 corrupted)
r = e['ANA-1']; etas = ['p0.5', 'p2', 'p10']; xl = ['η = 0.5', 'η = 2', 'η = 10']
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=250)
ax = axs[0]; ax.bar(xl, [r[k]['w'] for k in etas], color=C1, width=0.55)
ax.axhline(0.5, ls='--', color=C2, lw=1.3); ax.text(-0.3, 0.51, 'initial weight 0.5', fontsize=9, color='#555')
ax.set_ylabel('Mean trust in corrupted agent'); ax.set_xlabel('(a)'); ax.set_ylim(0, 0.6); ax.grid(True, axis='y', alpha=0.3)
ax = axs[1]; ax.plot(xl, [100 * r[k]['hit'] for k in etas], '-o', color=C1, lw=1.8, label='Proposed')
ax.axhline(100 * r['mean']['hit'], ls='--', color=C2, lw=1.5, label='Simple averaging')
ax.axhline(100 * x7['k3_median']['ANA-1'][1], ls=':', color=C3, lw=1.8, label='Coordinate-wise median')
ax.axhline(100 * bc['k3']['ANA-1'][1], ls='-.', color=C4, lw=1.5, label='Centralized BI (rule (6))')
ax.set_ylabel('Best department found, %'); ax.set_xlabel('(b)'); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='lower right')
fig.tight_layout(w_pad=3); fig.savefig('fig4.png'); plt.close(fig)
print('ok')
