"""
Controlled synthetic experiments for
"Joint Convergence Analysis of Coupled Consensus and RL Dynamics in Multi-Agent MCDM".

Implements Eqs. (1)-(12) of the paper exactly:
  (1)/(4) consensus iteration  F(t+1) = ((1-a)I + a W(t)) F(t)
  (3)     exponential trust rule with lower bound w_min and renormalisation
  (5)-(6) Pareto filtering
  (7)-(10) consensus-aware Q-learning, epsilon-greedy over the Pareto set
  (12)    termination: max_{i,i'} d_{i,i'} < eps,  d_{i,i'} = max_r ||F_i(x_r) - F_i'(x_r)||_2

Run:  python sim.py   (numpy, matplotlib)   -> results.json + fig5.png + fig6.png
"""
import json, sys
import numpy as np

K, N, J = 50, 20, 5          # agents, variants, criteria
N_OUT = 5                    # persistently unreliable (outlier) agents
DEG = 10                     # neighbourhood size |N_i| (random 10-regular graph)
P = dict(alpha=0.3, eta=0.5, wmin=0.05, eps=0.01, mu=0.7, beta=0.1, gamma=0.9, rho=0.1)
SIGMA = 0.05                 # observation noise of reliable agents
T_CAP = 200                  # consensus iteration cap
SEEDS = 30                   # independent random seeds (L)


def regular_graph(rng, k=K, d=DEG):
    """Random d-regular graph built as the union of d edge-disjoint random perfect
    matchings (k even); such graphs are connected expanders with high probability."""
    while True:
        A = np.zeros((k, k), int)
        for _ in range(d):
            for _try in range(1000):
                p = rng.permutation(k).reshape(-1, 2)
                if not A[p[:, 0], p[:, 1]].any():
                    A[p[:, 0], p[:, 1]] = A[p[:, 1], p[:, 0]] = 1
                    break
        if (A.sum(1) == d).all():
            seen, stack = {0}, [0]
            while stack:
                u = stack.pop()
                for v in np.nonzero(A[u])[0]:
                    if v not in seen:
                        seen.add(v); stack.append(v)
            if len(seen) == k:
                return A


def pairwise_div(F):
    # d_{i,i'} = max_r ||F_i(x_r) - F_i'(x_r)||_inf   (dissertation (3.31))
    return np.abs(F[:, None] - F[None]).max(axis=(2, 3))


def consensus(F0, A, adaptive, alpha, eta, wmin, eps, cap=T_CAP, trace=False):
    deg = A.sum(1, keepdims=True)
    W = A / deg
    F = F0.copy()
    hist = []
    for t in range(cap + 1):
        D = pairwise_div(F)
        m = D.max()
        if trace:
            hist.append(m)
        if m < eps:
            return F, t, W, hist
        if adaptive:
            W = W * np.exp(-eta * D) * A
            W = W / W.sum(1, keepdims=True)
            n = A.sum(1, keepdims=True)
            W = ((1 - n * wmin) * W + wmin) * A          # affine floor (3.33)
        M = (1 - alpha) * np.eye(K) + alpha * W
        F = np.einsum('ik,knj->inj', M, F)
    return F, cap, W, hist


def make_env(rng):
    V = rng.uniform(0, 1, (N, J))                 # common (true) profile
    out_profile = rng.uniform(0, 1, (N_OUT, N, J))  # persistent outlier profiles
    omega = rng.dirichlet(np.ones(J))             # common criterion weights
    U = V @ omega
    return V, out_profile, omega, U


def observe(rng, V, out_profile):
    F = np.clip(V[None] + rng.normal(0, SIGMA, (K, N, J)), 0, 1)
    F[K - N_OUT:] = np.clip(out_profile + rng.normal(0, SIGMA, (N_OUT, N, J)), 0, 1)
    return F


def pareto(Fc):
    idx = []
    for a in range(N):
        dom = np.all(Fc >= Fc[a], 1) & np.any(Fc > Fc[a], 1)
        if not dom.any():
            idx.append(a)
    return np.array(idx)


# ------------------------------------------------------------------ Exp. A
def exp_convergence(seeds=SEEDS):
    res = {'adaptive': [], 'static': [], 'trace_adaptive': [], 'trace_static': []}
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        A = regular_graph(rng)
        V, O, _, _ = make_env(rng)
        F0 = observe(rng, V, O)
        for name, ad in (('adaptive', True), ('static', False)):
            _, it, _, h = consensus(F0, A, ad, P['alpha'], P['eta'], P['wmin'], P['eps'], trace=True)
            res[name].append(it)
            res['trace_' + name].append(h)
    return res


def exp_sensitivity(seeds=SEEDS):
    out = {'alpha': {}, 'eta': {}}
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        its = []
        for s in range(seeds):
            rng = np.random.default_rng(2000 + s)
            A = regular_graph(rng); V, O, _, _ = make_env(rng); F0 = observe(rng, V, O)
            its.append(consensus(F0, A, True, a, P['eta'], P['wmin'], P['eps'])[1])
        out['alpha'][a] = its
    for e in (0.1, 0.3, 0.5, 0.7, 1.0):
        its = []
        for s in range(seeds):
            rng = np.random.default_rng(3000 + s)
            A = regular_graph(rng); V, O, _, _ = make_env(rng); F0 = observe(rng, V, O)
            its.append(consensus(F0, A, True, P['alpha'], e, P['wmin'], P['eps'])[1])
        out['eta'][e] = its
    return out


# ------------------------------------------------------------------ Exp. B
EPOCHS, EVAL = 2000, 500     # learning epochs; accuracy over the last EVAL epochs


def run_learning(method, seed):
    """Closed-loop decision making on a repeated single-state decision task.
    Returns greedy-decision accuracy over the last EVAL epochs (true best = argmax U)."""
    rng = np.random.default_rng(5000 + seed)
    A = regular_graph(rng)
    V, O, omega, U = make_env(rng)
    best = int(np.argmax(U))
    beta, gamma, rho, mu = P['beta'], P['gamma'], P['rho'], P['mu']
    Q = np.full((K, N), 1.0 / (1 - gamma))   # optimistic initialisation (all methods)
    hits, its = [], []
    for t in range(EPOCHS):
        F0 = observe(rng, V, O)
        if method in ('proposed', 'static_consensus'):
            Fc, it, _, _ = consensus(F0, A, method == 'proposed', P['alpha'], P['eta'], P['wmin'], P['eps'])
            its.append(it)
            Fbar = Fc.mean(0)
            XP = pareto(Fbar)
            qbar = Q[:, XP].mean(0)
            greedy = XP[int(np.argmax(qbar))]
            x = greedy if rng.random() > rho else XP[rng.integers(len(XP))]
            Uc = Fbar[x] @ omega                        # utility under agreed evaluations
            resid = np.sqrt(((Fc - Fbar) ** 2).sum(-1)).max(-1)
            c = np.clip(1 - resid / np.sqrt(J), 0, 1)   # consensus level c_i in [0,1]
            R = mu * Uc + (1 - mu) * c                   # Eq. (7)
            td = R + gamma * Q[:, XP].max(1) - Q[:, x]   # Eq. (8)
            Q[:, x] += beta * td                         # Eq. (10)
        elif method == 'qd_learning':
            # consensus + innovations on Q-values (Kar et al. [16]); local rewards, no evaluation consensus
            qbar = Q.mean(0)
            greedy = int(np.argmax(qbar))
            x = greedy if rng.random() > rho else int(rng.integers(N))
            Rloc = F0[:, x, :] @ omega
            td = Rloc + gamma * Q.max(1) - Q[:, x]
            deg = A.sum(1, keepdims=True)
            cons = (A @ Q) - deg * Q                     # sum_j (Q_j - Q_i)
            Q = Q + 0.05 * cons
            Q[:, x] += beta * td
        elif method == 'independent':
            # independent Q-learners, plurality vote of greedy actions
            acts = np.where(rng.random(K) > rho, Q.argmax(1), rng.integers(N, size=K))
            Rloc = np.einsum('kj,j->k', F0[np.arange(K), acts, :], omega)
            td = Rloc + gamma * Q.max(1) - Q[np.arange(K), acts]
            Q[np.arange(K), acts] += beta * td
            greedy = int(np.bincount(Q.argmax(1), minlength=N).argmax())
        if t >= EPOCHS - EVAL:
            hits.append(greedy == best)
    regret = None
    return float(np.mean(hits)), (float(np.mean(its)) if its else None)


def exp_baselines(seeds=SEEDS):
    out = {}
    for m in ('proposed', 'static_consensus', 'qd_learning', 'independent'):
        acc, it = [], []
        for s in range(seeds):
            a, i = run_learning(m, s)
            acc.append(a); it.append(i)
        out[m] = {'acc': acc, 'iters': it}
        print(m, np.mean(acc), np.std(acc, ddof=1), file=sys.stderr)
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    R = {}
    if which in ('all', 'A'):
        R['conv'] = exp_convergence()
        R['sens'] = exp_sensitivity()
    if which in ('all', 'B'):
        R['base'] = exp_baselines()
    json.dump(R, open(f'results_{which}.json', 'w'), default=float)
