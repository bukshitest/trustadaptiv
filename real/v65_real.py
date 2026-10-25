"""Re-implementation of the dissertation (v65) real-data experiment, Chapter V (§5.2–5.3).

Agents   : ANA-1 DSc/professor, ANA-2 PhD/dotsent, ANA-3 no degree   (K = 3, complete graph, |N_i| = 2)
Criteria : f1 research, f2 teaching-methodical, f3 international      (benefit criteria)
y_{i,j}(r): mean criterion points of group-i teachers in department r; missing -> agent's own median
Normalization (4.12): common min/max over agents and departments
Trust rule (4.18)-(4.19): exponential update + affine floor w = (1-|N_i| w_min) w~ + w_min, inf-norm divergence
Selection (3.37)/(4.4): Pareto set of collective evaluations; U(x) = 1 - max_j lambda_j (f*_j - f^c_j)
Test     : decision from 2025, quality from observed 2026 values (pooled department means)
"""
import json
import numpy as np, pandas as pd
from scipy import stats

CRIT = {
    'f1': ['Web of science bazasida indekslangan maqolalar', 'Maqola Scopus', 'Scopus h-indeks',
           'OAK jurnali maqolasi (respublika/xalqaro)', 'OAKda qayd etilmagan maqolalar (respublika/xalqaro)',
           'Konfrensiya materiallari (Tezis)', 'Monografiya', 'Patent', 'Dasturiy mahsulot (DGU)'],
    'f2': ["5 million sun'iy intellekt yetakchilari sertifikati", "O'quv adabiyotlari (darslik/qo'llanma)",
           'Kuratorlik faoliyati', 'Talaba yutuqlari', 'Bitiruvchilar bandligini monitoringi',
           "5 million sun'iy intellekt yetakchilari sertifikati (talaba)"],
    'f3': ['Xorijiy grantlar', "Xorijiy OTMda dars o'tish", 'Xorijiy til bilish darajasi',
           'Xorijiy OTM magistratura diplomi', "Chet tilida fan o'qitish",
           "QS World University Rankings so'rovnomada korxonalar hamda mahalliy va xorijiy professor-o'qituvchilarni jalb etish"],
}
CN = ['f1', 'f2', 'f3']
AG = ['ANA-1', 'ANA-2', 'ANA-3']
P = dict(alpha=0.3, eta=0.5, wmin=0.05, eps=0.01, tmax=100)
LAM = np.array([0.40, 0.35, 0.25])


def prep(path):
    d = pd.read_pickle(path)
    for c, cols in CRIT.items():
        d[c] = d[cols].sum(axis=1)
    d['g'] = np.select([d['has_Professor'] | d['has_DSc'], d['has_PhD'] | d['has_Dotsent']], AG[:2], AG[2])
    return d


def agent_y(d, depts):
    Y = np.full((3, len(depts), 3), np.nan)
    gm = d.groupby(['g', 'dept'])[CN].mean()
    for i, g in enumerate(AG):
        for r, dep in enumerate(depts):
            if (g, dep) in gm.index:
                Y[i, r] = gm.loc[(g, dep)].values
        med = np.nanmedian(Y[i], axis=0)
        miss = np.isnan(Y[i]).any(1)
        Y[i, miss] = med
    return Y


def norm(Y, axes):
    lo = np.nanmin(Y, axis=axes, keepdims=True); hi = np.nanmax(Y, axis=axes, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    out = (Y - lo) / rng
    return np.where(hi > lo, out, 1.0)


def consensus(F0, adaptive, alpha=P['alpha'], eta=P['eta'], wmin=P['wmin'], eps=P['eps'], tmax=P['tmax'], trace=False):
    K = F0.shape[0]; A = 1 - np.eye(K); n = A.sum(1, keepdims=True)
    W = A / n; F = F0.copy(); hist = []
    for t in range(tmax + 1):
        D = np.abs(F[:, None] - F[None]).max(axis=(2, 3))      # inf-norm, max over variants
        hist.append((D.max(), W.copy()))
        if D.max() < eps:
            return F, t, W, hist
        if adaptive:
            Wt = W * np.exp(-eta * D) * A
            Wt = Wt / Wt.sum(1, keepdims=True)
            W = ((1 - n * wmin) * Wt + wmin) * A                 # (4.19)
        M = (1 - alpha) * np.eye(K) + alpha * W
        F = np.einsum('ik,knj->inj', M, F)
    return F, tmax, W, hist


def pareto(X):
    n = len(X)
    return np.array([not any(np.all(X[b] >= X[a]) and np.any(X[b] > X[a]) for b in range(n) if b != a)
                     for a in range(n)])


def utility(X, lam=LAM):
    P_ = pareto(X)
    fstar = X[P_].max(0)
    return 1 - (lam * (fstar - X)).max(1), P_


def truth(d26, depts, lam=LAM):
    Yt = d26.groupby('dept')[CN].mean().reindex(depts).values
    Xt = norm(Yt, 0)
    Ut, Pt = utility(Xt, lam)
    best = int(np.argmax(np.where(Pt, Ut, -np.inf)))
    return Ut, Pt, best


def decide(d, depts, method, lam=LAM, eta=P['eta'], corrupt=None):
    if method == 'bi':
        X = norm(d.groupby('dept')[CN].mean().reindex(depts).values, 0)
        s = X @ lam
        return s, int(np.argmax(s)), None, None
    Y = agent_y(d, depts)
    F = norm(Y, (0, 1))
    if method == 'mean':
        Fc, T, W = F.mean(0), None, None
    else:
        Fk, T, W, _ = consensus(F, method == 'proposed', eta=eta)
        Fc = Fk.mean(0)
    U, Pm = utility(Fc, lam)
    x = int(np.argmax(np.where(Pm, U, -np.inf)))
    return U, x, T, W


def metrics(score, x, Ut, Pt, best):
    rs = stats.spearmanr(score, Ut).statistic
    top5 = len(set(np.argsort(-score)[:5]) & set(np.argsort(-Ut)[:5])) / 5
    return dict(rs=rs, hit=float(x == best), phit=float(Pt[x]), top5=top5, regret=float(Ut[best] - Ut[x]))


def bootstrap(d, rng):
    idx = np.concatenate([rng.choice(g.index.values, size=len(g), replace=True) for _, g in d.groupby(['dept', 'g'])])
    return d.loc[idx].reset_index(drop=True)


def main(B=200):
    d25, d26 = prep('kpi2025.pkl'), prep('kpi2026.pkl')
    depts = sorted(set(d25.dept) & set(d26.dept))
    Ut, Pt, best = truth(d26, depts)
    out = {'counts': d25.g.value_counts().to_dict(), 'N': len(depts),
           'table51': {g: {'n': int((d25.g == g).sum()), 'depts': int(d25[d25.g == g].dept.nunique()),
                           **{c: [round(d25[d25.g == g][c].mean(), 2), round(d26[d26.g == g][c].mean(), 2)] for c in CN}}
                       for g in AG}}
    M = ['proposed', 'static', 'mean', 'bi']
    orig = {}
    for m in M:
        s, x, T, W = decide(d25, depts, m)
        orig[m] = {**metrics(s, x, Ut, Pt, best), 'T': T, 'x': depts[x]}
    out['orig'] = orig
    Y = agent_y(d25, depts); F = norm(Y, (0, 1))
    Fk, T, W, hist = consensus(F, True)
    out['trace'] = [(float(h[0]), h[1].round(3).tolist()) for h in hist]
    U, Pm = utility(Fk.mean(0))
    order = np.argsort(-U)[:8]
    out['top8'] = [(depts[r], bool(Pm[r]), round(float(U[r]), 3), round(float(Ut[r]), 3),
                    int((-Ut).argsort().tolist().index(r)) + 1) for r in order]
    out['n_pareto'] = int(Pm.sum()); out['best_true'] = depts[best]
    res = {m: [] for m in M}
    for b in range(B):
        rng = np.random.default_rng(9000 + b)
        db = bootstrap(d25, rng)
        for m in M:
            s, x, T, W = decide(db, depts, m)
            res[m].append({**metrics(s, x, Ut, Pt, best), 'T': T})
    out['boot'] = {m: pd.DataFrame(v).to_dict('list') for m, v in res.items()}
    json.dump(out, open('v65_real.json', 'w'), default=lambda o: o if not isinstance(o, np.generic) else o.item())
    return out


if __name__ == '__main__':
    o = main()
    print('counts', o['counts']); print('table51', json.dumps(o['table51'], indent=0))
    print('ORIG'); [print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}) for m, r in o['orig'].items()]
    print('trace', [round(t[0], 4) for t in o['trace']])
    print('final W', o['trace'][-1][1])
    print('npareto', o['n_pareto'], 'best', o['best_true']); [print(r) for r in o['top8']]
    for m, v in o['boot'].items():
        df = pd.DataFrame(v)
        print('%-9s rs %.3f±%.3f hit %.0f%% top5 %.0f%% regret %.4f T %s' % (m, df.rs.mean(), df.rs.std(), 100 * df.hit.mean(),
              100 * df.top5.mean(), df.regret.mean(), round(df['T'].mean(), 1) if df['T'].notna().any() else '-'))
