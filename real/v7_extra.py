"""Additional analyses for v7 revision: (a) coordinate-wise median baseline; (b) K = 6 agents
(each degree group split at random into two halves) with one corrupted agent."""
import json, numpy as np, pandas as pd
from v65_real import *

def decide_median(d, depts):
    F = norm(agent_y(d, depts), (0, 1)); Fc = np.median(F, 0)
    U, Pm = utility(Fc); return U, int(np.argmax(np.where(Pm, U, -np.inf)))

def agent_y_split(d, depts, labels):
    ags = sorted(d[labels].unique()); Y = np.full((len(ags), len(depts), 3), np.nan)
    gm = d.groupby([labels, 'dept'])[CN].mean()
    for i, g in enumerate(ags):
        for r, dep in enumerate(depts):
            if (g, dep) in gm.index: Y[i, r] = gm.loc[(g, dep)].values
        Y[i, np.isnan(Y[i]).any(1)] = np.nanmedian(Y[i], axis=0)
    return Y, ags

d25, d26 = prep('kpi2025.pkl'), prep('kpi2026.pkl')
depts = sorted(set(d25.dept) & set(d26.dept)); Ut, Pt, best = truth(d26, depts)
out = {}
# (a) median on clean data: original + bootstrap
U, x = decide_median(d25, depts); out['median_orig'] = metrics(U, x, Ut, Pt, best)
bm = []
for b in range(200):
    db = bootstrap(d25, np.random.default_rng(9000 + b)); U, x = decide_median(db, depts); bm.append(metrics(U, x, Ut, Pt, best))
bm = pd.DataFrame(bm); out['median_boot'] = bm.to_dict('list')
o = json.load(open('v65_real.json')); prop = np.array(o['boot']['proposed']['rs'])
diff = prop - bm.rs.values
print('median orig', {k: round(float(v), 3) for k, v in out['median_orig'].items()})
print('median boot rs %.3f±%.3f hit %.1f top5 %.1f regret %.4f' % (bm.rs.mean(), bm.rs.std(), 100*bm.hit.mean(), 100*bm.top5.mean(), bm.regret.mean()))
print('prop-median %.3f [%.3f, %.3f] better %.0f%%' % (diff.mean(), np.quantile(diff, .025), np.quantile(diff, .975), 100*(diff > 1e-12).mean()))
# median under corruption, K = 3
def corrupt(d, agent, rng, col='g'):
    d = d.copy()
    for dep in d.dept.unique():
        m = (d.dept == dep) & (d[col] == agent); d.loc[m, CN] = d.loc[m, CN].values * rng.uniform(0, 3)
    return d
out['k3_median'] = {}
for ag in AG:
    r = []
    for rep in range(100):
        dc = corrupt(d25, ag, np.random.default_rng(11000 + 100 * AG.index(ag) + rep))
        U, x = decide_median(dc, depts); r.append(metrics(U, x, Ut, Pt, best))
    r = pd.DataFrame(r); out['k3_median'][ag] = (float(r.rs.mean()), float(r.hit.mean()))
    print('K3 median corrupt', ag, round(r.rs.mean(), 3), round(100 * r.hit.mean()))
# (b) K = 6
res = {m: [] for m in ('bi', 'mean', 'median', 'p0.5', 'p2', 'p10')}; wc = {k: [] for k in ('p0.5', 'p2', 'p10')}
for rep in range(100):
    rng = np.random.default_rng(30000 + rep)
    d = d25.copy(); half = rng.integers(0, 2, len(d)); d['g6'] = d['g'] + np.where(half == 0, 'a', 'b')
    target = AG[rep % 3] + 'a'
    dc = corrupt(d, target, rng, col='g6')
    s, x, *_ = decide(dc, depts, 'bi'); res['bi'].append(metrics(s, x, Ut, Pt, best))
    Y, ags = agent_y_split(dc, depts, 'g6'); F = norm(Y, (0, 1)); c = ags.index(target)
    U, Pm = utility(F.mean(0)); x = int(np.argmax(np.where(Pm, U, -np.inf))); res['mean'].append(metrics(U, x, Ut, Pt, best))
    U, Pm = utility(np.median(F, 0)); x = int(np.argmax(np.where(Pm, U, -np.inf))); res['median'].append(metrics(U, x, Ut, Pt, best))
    for eta, k in ((0.5, 'p0.5'), (2.0, 'p2'), (10.0, 'p10')):
        Fk, T, W, _ = consensus(F, True, eta=eta); U, Pm = utility(Fk.mean(0)); x = int(np.argmax(np.where(Pm, U, -np.inf)))
        res[k].append({**metrics(U, x, Ut, Pt, best), 'group': target[:-1]})
        wc[k].append(float(np.mean([W[i, c] for i in range(len(ags)) if i != c])))
out['k6'] = {}
for m, v in res.items():
    df = pd.DataFrame(v); out['k6'][m] = {'rs': float(df.rs.mean()), 'hit': float(df.hit.mean()), 'w': float(np.mean(wc[m])) if m in wc else None}
    if m in wc:
        out['k6'][m]['by_group'] = {g: (float(df[df.group == g].rs.mean()), float(df[df.group == g].hit.mean()), float(np.mean(np.array(wc[m])[df.group.values == g]))) for g in AG}
    print('K6', m, {k: (round(v2, 3) if isinstance(v2, float) else v2) for k, v2 in out['k6'][m].items()})
json.dump(out, open('v7_extra.json', 'w'), default=float)
