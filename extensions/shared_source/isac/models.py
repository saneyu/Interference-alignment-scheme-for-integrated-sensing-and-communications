"""Columns are transmitters; rows are time samples. h is never conjugated in y=Xh.

Transmit designs depend only on dimensions/topology, never instantaneous sensing CSI.
All power caps are expectations over independent zero-mean unit-power data symbols.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np


def cn(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def qpsk(rng, shape):
    bits = rng.integers(0, 2, size=(*shape, 2))
    return ((1-2*bits[..., 0]) + 1j*(1-2*bits[..., 1])) / np.sqrt(2), bits


def dft_rows(rows, cols):
    if cols < rows:
        raise ValueError('Need at least as many pilot observations as channel coefficients.')
    return np.exp(-2j*np.pi*np.outer(np.arange(rows), np.arange(cols))/cols) / np.sqrt(cols)


def left_null(a, tol=1e-11):
    u, s, _ = np.linalg.svd(a, full_matrices=True)
    rank = np.sum(s > tol * (s[0] if len(s) else 1.0))
    return u[:, rank:]  # orthonormal columns; U^H a=0


@dataclass
class Design:
    name: str
    block: int
    blocks: int
    pilot: np.ndarray       # T x n_tx
    comm: np.ndarray        # T x n_tx x n_symbols
    q: np.ndarray           # T x n_projected, semiunitary
    sensed: np.ndarray
    alpha: float
    difference: np.ndarray | None = None
    topology: np.ndarray | None = None
    vectors: np.ndarray | None = None

    @property
    def T(self):
        return self.block * self.blocks

    @property
    def n_tx(self):
        return self.pilot.shape[1]

    @property
    def expected_power(self):
        return abs(self.pilot)**2 + np.sum(abs(self.comm)**2, axis=2)


def _scale_pilots(p, alpha):
    peak = np.max(abs(p)**2, axis=0)
    return p * np.sqrt(alpha / np.where(peak > 0, peak, 1))[None, :]


def bia(k=3, blocks=2, alpha=.5, lift='paper'):
    """Eq. (12), or minimum-energy orthogonal lift with the same nulling structure."""
    if k < 2 or blocks*(k-1) < k or not 0 < alpha < 1:
        raise ValueError('Require K>=2, B(K-1)>=K, 0<alpha<1.')
    if lift not in ('paper', 'orthogonal'):
        raise ValueError('Unknown pilot lift.')
    f = dft_rows(k, blocks*(k-1))
    u = left_null(np.ones((k, 1)))
    a = np.ones((k, k-1)); a[1:] -= np.eye(k-1)
    p = np.zeros((blocks*k, k), complex)
    c = np.zeros((blocks*k, k, blocks*k), complex)
    q = np.kron(np.eye(blocks), u)
    d0 = np.column_stack((np.ones(k-1), -np.eye(k-1)))
    for b in range(blocks):
        sl = slice(b*k, (b+1)*k)
        p[sl] = (a if lift == 'paper' else u) @ f[:, b*(k-1):(b+1)*(k-1)].T
        for j in range(k):
            c[sl, j, b*k+j] = np.sqrt(1-alpha)
    return Design('BIA', k, blocks, _scale_pilots(p, alpha), c, q,
                  np.arange(k), alpha, np.kron(np.eye(blocks), d0))


def tim(blocks=2, alpha=.5):
    """Exact (6,1,2) Example 2; fixed full-spark DFT frame replaces random vectors.

    Sensor sees TX 3,4,6 (1-based), communication spans v3,v4,v5.
    The v2,v3,v6 nullspace phrase on printed p86 is inconsistent with its equation.
    """
    if blocks < 2 or not 0 < alpha < 1:
        raise ValueError('Require >=2 blocks and 0<alpha<1.')
    n, t = 6, 5
    v = np.exp(-2j*np.pi*np.outer(np.arange(t), np.arange(n))/n) / np.sqrt(t)
    topology = np.ones((n, n), dtype=bool)
    for r in range(n):
        topology[r, [(r-1)%n, (r+1)%n, (r+2)%n]] = False
    sensed = np.flatnonzero(topology[-1])
    u = left_null(v[:, [2, 3, 4]])
    f = dft_rows(3, blocks*2)
    p = np.zeros((t*blocks, n), complex)
    c = np.zeros((t*blocks, n, blocks*10), complex)
    for b in range(blocks):
        sl = slice(t*b, t*(b+1))
        p[sl, sensed] = u @ f[:, 2*b:2*b+2].T
        for j in range(5):
            c[sl, j, b*10+2*j:b*10+2*j+2] = v[:, j:j+2] * np.sqrt((1-alpha)*t/2)
    return Design('TIM', t, blocks, _scale_pilots(p, alpha), c,
                  np.kron(np.eye(blocks), u), sensed, alpha,
                  topology=topology, vectors=v)


def operators(design, sigma2, prior=None):
    """Receive maps E such that hhat = E y. Linear solves/pinv only."""
    p = design.pilot[:, design.sensed]
    q = design.q
    a = q.conj().T @ p
    result = {'Projected LS': np.linalg.pinv(a) @ q.conj().T,
              'TIN (same waveform)': np.linalg.pinv(p)}
    if design.difference is not None:
        d = design.difference
        result['Difference OLS'] = np.linalg.pinv(d @ p) @ d
        w = np.linalg.solve(np.linalg.cholesky(d @ d.conj().T), d)
        result['Difference GLS'] = np.linalg.pinv(w @ p) @ w
    r = np.eye(len(design.sensed)) if prior is None else np.asarray(prior)
    result['Projected LMMSE'] = np.linalg.solve(
        a @ r @ a.conj().T + sigma2*np.eye(a.shape[0]), a @ r
    ).conj().T @ q.conj().T
    return result


def decision_directed_bia(design, y, iterations=4):
    """Feasible finite-alphabet benchmark: joint detection / LS alternation.

    Initializes ONLY from TIN LS; enumerates 4^K mixtures, never individual
    DQPSK-demodulates a multiuser mixture. Local iterations have no global guarantee.
    This is a new joint decision-directed baseline, not the paper's SIC decoder.
    """
    k = design.n_tx
    if design.name != 'BIA' or k > 4:
        raise ValueError('Joint-DD implemented only for BIA K<=4.')
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
    candidates = np.array(list(product(constellation, repeat=k)))
    hhat = y @ np.linalg.pinv(design.pilot).T
    for _ in range(iterations):
        residual = y - hhat @ design.pilot.T
        z = residual.reshape(len(y), design.blocks, k).mean(axis=2)
        mixtures = np.sqrt(1-design.alpha) * (hhat @ candidates.T)
        idx = np.argmin(abs(z[:, :, None]-mixtures[:, None, :])**2, axis=2)
        detected = candidates[idx].reshape(len(y), -1)
        data = np.einsum('tjs,ns->ntj', design.comm, detected, optimize=True)
        x = design.pilot[None] + data
        hhat = np.einsum('nmt,nt->nm', np.linalg.pinv(x), y)
    return hhat


def sensing(design, snr_db, trials, seed, drift=0., leakage=0., prior=None, dd=False):
    """Independent channel realizations; paired data/noise for all estimators.

    drift: std. per-link frequency in cycles/slot; estimate channel at frame center.
    leakage: power gain of nominally absent sensor links (TIM only).
    """
    rng = np.random.default_rng(seed)
    m = len(design.sensed)
    r = np.eye(m) if prior is None else np.asarray(prior)
    h = cn(rng, (trials, m)) @ np.linalg.cholesky(r).T
    sigma2 = 10**(-snr_db/10)
    symbols, _ = qpsk(rng, (trials, design.comm.shape[2]))
    data = np.einsum('tjs,ns->ntj', design.comm, symbols, optimize=True)
    h_full = np.zeros((trials, design.n_tx), complex)
    h_full[:, design.sensed] = h
    absent = np.setdiff1d(np.arange(design.n_tx), design.sensed)
    if leakage and len(absent):
        h_full[:, absent] = np.sqrt(leakage)*cn(rng, (trials, len(absent)))
    freq = drift*rng.standard_normal((trials, design.n_tx))
    time = np.arange(design.T)-(design.T-1)/2
    ht = h_full[:, None, :] * np.exp(2j*np.pi*time[None, :, None]*freq[:, None, :])
    total_x = design.pilot[None, :, :] + data
    clean = np.sum(total_x*ht, axis=2)
    noise = np.sqrt(sigma2)*cn(rng, clean.shape)
    y = clean + noise
    ops = operators(design, sigma2, r)
    estimates = {name: y @ e.T for name, e in ops.items()}
    if dd:
        estimates['Joint-DD (init TIN)'] = decision_directed_bia(design, y)
    # Oracle knows all active payloads, but never uses true h in its estimator.
    # Under topology leakage this oracle still estimates the nominal channels only.
    oracle = np.linalg.pinv(total_x[:, :, design.sensed])
    estimates['Known-data oracle'] = np.einsum('nmt,nt->nm', oracle, y)
    norm = float(np.trace(r).real)
    error = {name: np.sum(abs(hh-h)**2, axis=1)/norm for name, hh in estimates.items()}
    cee = {name: e*norm / np.sum(abs(h)**2, axis=1) for name, e in error.items()}
    e = ops['Projected LS']
    theory = sigma2 * np.sum(abs(e)**2)/norm
    result = {'errors': error, 'cee': cee, 'theory': float(theory),
              'power_peak': float(design.expected_power.max()),
              'power_mean': float(design.expected_power.mean())}
    return result


def bia_communication(k, alpha, snr_db, trials, seed, rho=0., csir_nmse=0.):
    """K-user IC, K time slots, each Rx keeps ONLY its own stream.

    AR(1) coefficient rho controls slot-to-slot communication channel correlation.
    CSIR errors affect pilot cancellation as well as detection. No coding overhead.
    """
    if not 0 <= rho <= 1 or csir_nmse < 0:
        raise ValueError('Invalid channel/CSIR setting.')
    rng = np.random.default_rng(seed)
    s, bits = qpsk(rng, (trials, k))
    h = cn(rng, (trials, k, k, k))  # realization, receiver, time, transmitter
    for t in range(1, k):
        h[:, :, t, :] = rho*h[:, :, t-1, :] + np.sqrt(1-rho*rho)*h[:, :, t, :]
    he = h + np.sqrt(csir_nmse)*cn(rng, h.shape)
    p = bia(k, 2, alpha).pilot[:k]
    a, ae = np.sqrt(1-alpha)*h, np.sqrt(1-alpha)*he
    sigma2 = 10**(-snr_db/10)
    noise = np.sqrt(sigma2)*cn(rng, (trials, k, k))
    residual_pilot = np.einsum('nrti,ti->nrt', h-he, p)
    y = np.einsum('nrti,ni->nrt', a, s) + residual_pilot + noise
    out = {}
    for method in ('ZF', 'MMSE'):
        if method == 'ZF':
            w = np.linalg.pinv(ae)
        else:
            ah = ae.conj().swapaxes(-1, -2)
            w = np.linalg.solve(ah@ae + sigma2*np.eye(k), ah)
        own_w = w[:, np.arange(k), np.arange(k), :]
        sh = np.einsum('nrt,nrt->nr', own_w, y)
        gain = np.einsum('nrt,nrti->nri', own_w, a)
        desired = gain[:, np.arange(k), np.arange(k)]
        # Receiver equalizes using its channel estimate, not true channel.
        gain_est = np.einsum('nrt,nrti->nri', own_w, ae)
        eq_gain = gain_est[:, np.arange(k), np.arange(k)]
        sh = sh/np.where(abs(eq_gain)>1e-14, eq_gain, 1e-14)
        bh = np.stack((sh.real < 0, sh.imag < 0), axis=-1)
        ber = np.mean(bh != bits, axis=(1, 2))
        interf = np.maximum(np.sum(abs(gain)**2, axis=-1)-abs(desired)**2, 0)
        noise_out = sigma2*np.sum(abs(own_w)**2, axis=-1)
        # Used as rate only in perfect-CSIR sweeps; avoids claiming mismatched capacity.
        sinr = abs(desired)**2 / np.maximum(interf+noise_out, 1e-30)
        out[method] = {'ber': ber, 'rate': np.sum(np.log2(1+sinr), axis=1)/k}
    out['condition'] = np.linalg.cond(a[:, 0])
    return out


def tim_communication(alpha, snr_db, trials, seed):
    """Example 2, perfect CSIR, topology-only transmit vectors, 2 desired streams/Rx."""
    rng = np.random.default_rng(seed)
    design = tim(2, alpha)
    v, top = design.vectors, design.topology
    s, bits = qpsk(rng, (trials, 5, 2))
    h = cn(rng, (trials, 5, 6)) * top[None, :5, :]
    sigma2 = 10**(-snr_db/10)
    c = design.comm[:5, :, :10]
    data = np.einsum('tjs,ns->ntj', c, s.reshape(trials, 10))
    y = np.einsum('nrj,ntj->nrt', h, data) + np.sqrt(sigma2)*cn(rng, (trials, 5, 5))
    errs = np.zeros((trials, 5, 2, 2), bool)
    rates = np.zeros(trials)
    for j in range(5):
        others = [i for i in np.flatnonzero(top[j, :5]) if i != j]
        ids = sorted(set(i+offset for i in others for offset in (0, 1)))
        q = left_null(v[:, ids])
        a = h[:, j, j, None, None] * (q.conj().T @ c[:, j, 2*j:2*j+2])[None]
        w = np.linalg.pinv(a)
        z = y[:, j] @ q.conj()
        sh = np.einsum('nij,nj->ni', w, z)
        bh = np.stack((sh.real < 0, sh.imag < 0), axis=-1)
        errs[:, j] = bh != bits[:, j]
        # Separate-stream ZF achievable rate, same receiver convention as BIA.
        sinr = 1/(sigma2*np.sum(abs(w)**2, axis=-1))
        rates += np.sum(np.log2(1+sinr), axis=1)/5
    return {'ber': errs.mean(axis=(1, 2, 3)), 'rate': rates}
