import json, numpy as np, pandas as pd
from v65_real import *

def corrupt(d, agent, rng):
    d = d.copy()
    for dep in d.dept.unique():
        m = (d.dept == dep) & (d.g == agent)
        d.loc[m, CN] = d.loc[m, CN].values * rng.uniform(0, 3)
    return d

def decide_w(d, depts, eta, corrupted):
    Y = agent_y(d, depts); F = norm(Y, (0, 1))
    Fk, T, W, _ = consensus(F, True, eta=eta)
    U, Pm = utility(Fk.mean(0)); x = int(np.argmax(np.where(Pm, U, -np.inf)))
    c = AG.index(corrupted); wc = np.mean([W[i, c] for i in range(3) if i != c])
    return U, x, wc

d25, d26 = prep('kpi2025.pkl'), prep('kpi2026.pkl')
depts = sorted(set(d25.dept) & set(d26.dept))
Ut, Pt, best = truth(d26, depts)
out = {}
for ag in AG:
    rows = {'bi': [], 'mean': [], 'static': [], 'p0.5': [], 'p2': [], 'p10': []}
    for rep in range(100):
        rng = np.random.default_rng(11000 + 100 * AG.index(ag) + rep)
        dc = corrupt(d25, ag, rng)
        for m in ('bi', 'mean', 'static'):
            s, x, T, W = decide(dc, depts, m)
            rows[m].append({**metrics(s, x, Ut, Pt, best), 'w': None})
        for eta, key in ((0.5, 'p0.5'), (2.0, 'p2'), (10.0, 'p10')):
            U, x, wc = decide_w(dc, depts, eta, ag)
            rows[key].append({**metrics(U, x, Ut, Pt, best), 'w': wc})
    out[ag] = {k: {'rs': float(np.mean([r['rs'] for r in v])), 'hit': float(np.mean([r['hit'] for r in v])),
                   'w': (float(np.mean([r['w'] for r in v])) if v[0]['w'] is not None else None)} for k, v in rows.items()}
    print(ag, {k: (round(v['rs'], 3), round(100 * v['hit']), None if v['w'] is None else round(v['w'], 3)) for k, v in out[ag].items()}, flush=True)
sens = []
for lam in [(0.40, 0.35, 0.25), (1/3, 1/3, 1/3), (0.60, 0.25, 0.15), (0.25, 0.50, 0.25), (0.25, 0.25, 0.50)]:
    lam = np.array(lam); Ut2, Pt2, best2 = truth(d26, depts, lam)
    s, x, *_ = decide(d25, depts, 'proposed', lam=lam); sb, xb, *_ = decide(d25, depts, 'bi', lam=lam)
    r = (lam.round(2).tolist(), round(stats.spearmanr(s, Ut2).statistic, 3), round(stats.spearmanr(sb, Ut2).statistic, 3), x == best2, xb == best2)
    sens.append(r); print(r)
out['sens'] = sens
json.dump(out, open('v65_extra.json', 'w'), default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
