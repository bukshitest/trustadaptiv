"""Synthetic test of evaluation-mode adaptation by Q-learning (dissertation §3.2, §4.6; Eqs. (3.21)-(3.26)).

Data: departments x semesters x groups. True value v[t,r,j] = level + trend*t + season(t) + random walk.
Agent i (group) observes v + group bias drift + group noise. Mode a = (window, aggregation):
windows {2, 4, all} x aggregations {mean, EWMA, linear-trend forecast} = 9 modes (as in the dissertation).
Cycle tau: agents choose modes (rho-greedy), evaluate -> normalize (4.12) -> trust-adaptive consensus
(inf-norm, affine floor) -> Pareto + U(x) (3.37) -> x*; quality xi = U_true(x*) in semester tau+1;
reward R_i = mu*xi + (1-mu)*kappa_i (3.23)-(3.24); state = (kappa level, data-change level); Q-update (3.21)-(3.22).
"""
import json, sys
import numpy as np
from scipy import stats

K, N, J = 3, 33, 3
S_TOT, WARM = 60, 4                 # semesters; first decision after WARM semesters
WINDOWS = [2, 4, None]; AGGS = ['mean', 'ewma', 'trend']
MODES = [(w, g) for w in WINDOWS for g in AGGS]
P = dict(alpha=0.3, eta=0.5, wmin=0.05, eps=0.01, tmax=100, beta=0.2, gamma=0.8, rho=0.1, mu=0.6)
LAM = np.array([0.40, 0.35, 0.25])


def gen(rng):
    lvl = rng.uniform(0.2, 0.8, (N, J))
    t = np.arange(S_TOT)[:, None, None]
    # piecewise trends: slope changes at a random semester per department/criterion
    k = rng.integers(10, S_TOT - 10, (N, J))
    s1 = rng.normal(0, 0.006, (N, J)); s2 = rng.normal(0, 0.006, (N, J))
    trend = np.where(t < k, s1 * t, s1 * k + s2 * (t - k))
    season = 0.03 * np.where(np.arange(S_TOT)[:, None, None] % 2 == 0, 1, -1) * rng.uniform(-1, 1, (1, N, J))
    walk = np.cumsum(rng.normal(0, 0.01, (S_TOT, N, J)), axis=0)
    v = lvl + trend + season + walk
    bias = rng.normal(0, 0.05, (K, 1, 1, J)); drift = rng.normal(0, 0.002, (K, 1, 1, J))
    noise = np.array([0.08, 0.04, 0.03])[:, None, None, None]          # small groups are noisier
    obs = v[None] + bias + drift * np.arange(S_TOT)[None, :, None, None] + rng.normal(0, 1, (K, S_TOT, N, J)) * noise
    return v, obs


def evaluate(series, mode):
    """series: (T, N, J) history up to current semester -> forecast for next semester."""
    w, g = mode
    h = series if w is None else series[-w:]
    if g == 'mean':
        return h.mean(0)
    if g == 'ewma':
        wts = 0.5 ** np.arange(len(h))[::-1]; return (h * wts[:, None, None]).sum(0) / wts.sum()
    if len(h) < 2:
        return h[-1]
    x = np.arange(len(h)); xm = x.mean()
    slope = ((x - xm)[:, None, None] * (h - h.mean(0))).sum(0) / ((x - xm) ** 2).sum()
    return h.mean(0) + slope * (len(h) - xm)


def norm(Y, axes):
    lo = Y.min(axis=axes, keepdims=True); hi = Y.max(axis=axes, keepdims=True)
    return np.where(hi > lo, (Y - lo) / np.where(hi > lo, hi - lo, 1), 1.0)


def consensus(F0):
    A = 1 - np.eye(K); n = A.sum(1, keepdims=True); W = A / n; F = F0.copy()
    for t in range(P['tmax'] + 1):
        D = np.abs(F[:, None] - F[None]).max(axis=(2, 3))
        if D.max() < P['eps']:
            return F, t
        Wt = W * np.exp(-P['eta'] * D) * A; Wt /= Wt.sum(1, keepdims=True)
        W = ((1 - n * P['wmin']) * Wt + P['wmin']) * A
        F = np.einsum('ik,knj->inj', (1 - P['alpha']) * np.eye(K) + P['alpha'] * W, F)
    return F, P['tmax']


def pareto(X):
    return np.array([not np.any(np.all(X >= X[a], 1) & np.any(X > X[a], 1)) for a in range(len(X))])


def utility(X):
    Pm = pareto(X); fs = X[Pm].max(0)
    return 1 - (LAM * (fs - X)).max(1), Pm


def run(seed, policy):
    rng = np.random.default_rng(seed)
    v, obs = gen(rng)
    prng = np.random.default_rng(seed + 777)
    nS = 6
    Q = np.zeros((K, nS, len(MODES)))
    s = np.zeros(K, int); fixed = MODES.index((None, 'mean'))
    log = []
    for tau in range(WARM, S_TOT - 1):
        if policy == 'q':
            a = np.array([prng.integers(len(MODES)) if prng.random() < P['rho'] else int(np.argmax(Q[i, s[i]]))
                          for i in range(K)])
        elif policy == 'random':
            a = prng.integers(len(MODES), size=K)
        elif isinstance(policy, int):
            a = np.full(K, policy)
        else:
            a = np.full(K, fixed)
        Y = np.stack([evaluate(obs[i, :tau + 1], MODES[a[i]]) for i in range(K)])
        F0 = norm(Y, (0, 1)); Fk, T = consensus(F0); Fc = Fk.mean(0)
        U, Pm = utility(Fc); x = int(np.argmax(np.where(Pm, U, -np.inf)))
        Ut, Pt = utility(norm(v[tau + 1], 0)); best = int(np.argmax(np.where(Pt, Ut, -np.inf)))
        xi = Ut[x]
        kappa = 1 - np.abs(F0 - Fc[None]).mean(axis=(1, 2))
        change = np.abs(obs[:, tau] - obs[:, tau - 1]).mean(axis=(1, 2))
        s_next = np.digitize(kappa, [0.9, 0.95]) * 2 + (change > 0.05).astype(int)
        if policy == 'q':
            R = P['mu'] * xi + (1 - P['mu']) * kappa
            for i in range(K):
                td = R[i] + P['gamma'] * Q[i, s_next[i]].max() - Q[i, s[i], a[i]]
                Q[i, s[i], a[i]] += P['beta'] * td
        greedy = [int(np.argmax(Q[i, s_next[i]])) for i in range(K)]
        log.append(dict(regret=float(Ut[best] - Ut[x]), hit=float(x == best),
                        rs=float(stats.spearmanr(U, Ut).statistic), T=T, greedy=greedy, a=a.tolist()))
        s = s_next
    return log


def summarize(logs, last=None):
    out = {}
    for k in ('regret', 'hit', 'rs', 'T'):
        vals = [np.mean([c[k] for c in (lg if last is None else lg[-last:])]) for lg in logs]
        out[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    return out


if __name__ == '__main__':
    SEEDS = range(30)
    res = {}
    for pol in ('q', 'fixed', 'random'):
        res[pol] = [run(20000 + sd, pol) for sd in SEEDS]
        print(pol, summarize(res[pol], last=25), flush=True)
    # best fixed mode in hindsight (oracle reference)
    best_mode = []
    for m in range(len(MODES)):
        lg = [run(20000 + sd, m) for sd in SEEDS]
        best_mode.append((summarize(lg, last=25)['regret'][0], m, lg))
    best_mode.sort(key=lambda z: z[0])
    res['oracle'] = best_mode[0][2]
    print('oracle mode', MODES[best_mode[0][1]], summarize(res['oracle'], last=25))
    print('all fixed modes regret', [(MODES[m], round(r, 4)) for r, m, _ in best_mode])
    # policy stabilisation: share of greedy-action changes in consecutive cycles, first vs last 15 cycles
    ch = lambda lg, sl: np.mean([np.mean([g1 != g0 for g0, g1 in zip(lg[t - 1]['greedy'], lg[t]['greedy'])])
                                 for t in range(1, len(lg))][sl])
    early = [ch(lg, slice(0, 15)) for lg in res['q']]; late = [ch(lg, slice(-15, None)) for lg in res['q']]
    print('greedy change early %.3f late %.3f' % (np.mean(early), np.mean(late)))
    # paired tests on last 25 cycles regret
    r = lambda pol: np.array([np.mean([c['regret'] for c in lg[-25:]]) for lg in res[pol]])
    for b in ('fixed', 'random', 'oracle'):
        print('q vs', b, stats.wilcoxon(r('q'), r(b)).pvalue, (r('q') < r(b)).sum(), (r('q') > r(b)).sum())
    json.dump({p: [[{k: c[k] for k in ('regret', 'hit', 'rs', 'T')} for c in lg] for lg in v] for p, v in res.items()} |
              {'early': early, 'late': late, 'oracle_mode': str(MODES[best_mode[0][1]]),
               'fixed_modes': [(str(MODES[m]), r) for r, m, _ in best_mode]},
              open('results_rl.json', 'w'))
