import json, numpy as np, pandas as pd
from v65_real import *
def bi_cheb(d, depts):
    X = norm(d.groupby('dept')[CN].mean().reindex(depts).values, 0)
    U, Pm = utility(X); return U, int(np.argmax(np.where(Pm, U, -np.inf)))
d25, d26 = prep('kpi2025.pkl'), prep('kpi2026.pkl')
depts = sorted(set(d25.dept) & set(d26.dept)); Ut, Pt, best = truth(d26, depts)
out = {}
U, x = bi_cheb(d25, depts); out['orig'] = metrics(U, x, Ut, Pt, best); print('orig', out['orig'])
b = [metrics(*bi_cheb(bootstrap(d25, np.random.default_rng(9000 + i)), depts), Ut, Pt, best) for i in range(200)]
b = pd.DataFrame(b); out['boot'] = b.to_dict('list')
prop = np.array(json.load(open('v65_real.json'))['boot']['proposed']['rs']); diff = prop - b.rs.values
print('boot rs %.3f±%.3f hit %.1f top5 %.1f regret %.4f' % (b.rs.mean(), b.rs.std(), 100*b.hit.mean(), 100*b.top5.mean(), b.regret.mean()))
print('prop-bicheb %.3f [%.3f, %.3f] prop better %.1f%%' % (diff.mean(), np.quantile(diff, .025), np.quantile(diff, .975), 100*(diff > 1e-12).mean()))
out['diff'] = [float(diff.mean()), float(np.quantile(diff, .025)), float(np.quantile(diff, .975)), float((diff > 1e-12).mean())]
def corrupt(d, agent, rng, col='g'):
    d = d.copy()
    for dep in d.dept.unique():
        m = (d.dept == dep) & (d[col] == agent); d.loc[m, CN] = d.loc[m, CN].values * rng.uniform(0, 3)
    return d
out['k3'] = {}
for ag in AG:
    r = [metrics(*bi_cheb(corrupt(d25, ag, np.random.default_rng(11000 + 100 * AG.index(ag) + rep)), depts), Ut, Pt, best) for rep in range(100)]
    r = pd.DataFrame(r); out['k3'][ag] = (float(r.rs.mean()), float(r.hit.mean())); print('K3', ag, round(r.rs.mean(), 3), round(100 * r.hit.mean()))
r = []
for rep in range(100):
    rng = np.random.default_rng(30000 + rep); d = d25.copy(); half = rng.integers(0, 2, len(d)); d['g6'] = d['g'] + np.where(half == 0, 'a', 'b')
    dc = corrupt(d, AG[rep % 3] + 'a', rng, col='g6'); r.append(metrics(*bi_cheb(dc, depts), Ut, Pt, best))
r = pd.DataFrame(r); out['k6'] = (float(r.rs.mean()), float(r.hit.mean())); print('K6', out['k6'])
# sensitivity to lambda
sens = []
for lam in [(0.40, 0.35, 0.25), (1/3, 1/3, 1/3), (0.60, 0.25, 0.15), (0.25, 0.50, 0.25), (0.25, 0.25, 0.50)]:
    lam = np.array(lam); Ut2, Pt2, best2 = truth(d26, depts, lam)
    X = norm(d25.groupby('dept')[CN].mean().reindex(depts).values, 0); U, Pm = utility(X, lam); x = int(np.argmax(np.where(Pm, U, -np.inf)))
    s, xp, *_ = decide(d25, depts, 'proposed', lam=lam)
    sens.append((lam.round(2).tolist(), round(stats.spearmanr(s, Ut2).statistic, 3), round(stats.spearmanr(U, Ut2).statistic, 3), bool(xp == best2), bool(x == best2)))
    print(sens[-1])
out['sens'] = sens
json.dump(out, open('v8_bicheb.json', 'w'), default=float)
