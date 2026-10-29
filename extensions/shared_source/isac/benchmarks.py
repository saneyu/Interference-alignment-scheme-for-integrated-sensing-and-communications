"""Unaligned superposition and actual successive interference cancellation.

Sensing-only resource comparison: equal component energy per TX and same T.
Does NOT establish equal communication rate or BER across transmit designs.
"""
import numpy as np
from .models import cn, qpsk, operators


def unaligned_parameters(design):
    ids = design.sensed
    pilot_energy = np.sum(abs(design.pilot[:, ids])**2, axis=0)
    comm_power = np.sum(abs(design.comm[:, ids])**2, axis=(0, 2))/design.T
    t = np.arange(design.T)
    p = np.exp(2j*np.pi*np.outer(t, np.arange(len(ids)))/design.T)
    p *= np.sqrt(pilot_energy/design.T)
    return p, comm_power


def sic_symbols(y_data, h_est, comm_power):
    """Hard QPSK SIC, descending estimated received power, independently each slot."""
    n, t = y_data.shape
    active = np.flatnonzero(comm_power > 1e-15)
    residual = y_data.copy()
    decoded = np.zeros((n, t, len(comm_power)), complex)
    if not len(active):
        return decoded
    order = active[np.argsort(-abs(h_est[:, active])**2*comm_power[active], axis=1)]
    rows = np.arange(n)
    for stage in range(len(active)):
        j = order[:, stage]
        effective = h_est[rows, j]*np.sqrt(comm_power[j])
        soft = residual*np.conj(effective[:, None])
        hard = ((1-2*(soft.real < 0)) + 1j*(1-2*(soft.imag < 0)))/np.sqrt(2)
        decoded[rows, :, j] = hard*np.sqrt(comm_power[j])[:, None]
        residual -= effective[:, None]*hard
    return decoded


def sic_estimate(y, pilot, comm_power, iterations=4):
    if iterations < 1:
        raise ValueError('SIC requires at least one pass.')
    h_est = y@np.linalg.pinv(pilot).T
    for _ in range(iterations):
        y_data = y-h_est@pilot.T
        decoded = sic_symbols(y_data, h_est, comm_power)
        reconstructed = pilot[None]+decoded
        h_est = np.einsum('nmt,nt->nm', np.linalg.pinv(reconstructed), y)
    return h_est, decoded


def compare(design, snr_db, trials, seed, iterations=4):
    # Separate random streams so comparisons can share channel/noise without
    # depending on the number of symbols in either transmit design.
    children = np.random.SeedSequence(seed).spawn(4)
    rng_h, rng_n, rng_a, rng_b = [np.random.default_rng(s) for s in children]
    ids = design.sensed
    m, t = len(ids), design.T
    h = cn(rng_h, (trials, m))
    sigma2 = 10**(-snr_db/10)
    noise = np.sqrt(sigma2)*cn(rng_n, (trials, t))
    p, pc = unaligned_parameters(design)
    sb, bits = qpsk(rng_b, (trials, t, m))
    xb = sb*np.sqrt(pc)[None, None, :]
    yb = np.einsum('ntm,nm->nt', p[None]+xb, h)+noise
    ht = yb@np.linalg.pinv(p).T
    hs, detected = sic_estimate(yb, p, pc, iterations)
    sa, _ = qpsk(rng_a, (trials, design.comm.shape[2]))
    xa = np.einsum('tms,ns->ntm', design.comm[:, ids], sa, optimize=True)
    ya = np.einsum('ntm,nm->nt', design.pilot[:,ids][None]+xa, h)+noise
    hp = ya@operators(design, sigma2)['Projected LS'].T
    estimates = {design.name: hp, 'SIC': hs, 'TIN': ht}
    err = {name: np.sum(abs(estimate-h)**2, axis=1) for name,estimate in estimates.items()}
    cee = {name:e/np.sum(abs(h)**2,axis=1) for name,e in err.items()}
    nmse = {name:e/m for name,e in err.items()}
    active = np.flatnonzero(pc > 1e-15)
    bh = np.stack((detected.real<0,detected.imag<0),axis=-1)
    ber = np.mean(bh[:,:,active]!=bits[:,:,active],axis=(1,2,3))
    # Analytic TIN risk: independent per-slot data -> white unconditional covariance.
    pilot_inverse_energy = np.sum(abs(np.linalg.pinv(p))**2)/m
    theory = {'TIN': float((sigma2+pc.sum())*pilot_inverse_energy),
              'TIN_floor': float(pc.sum()*pilot_inverse_energy),
              design.name: float(sigma2*np.sum(abs(operators(design,sigma2)['Projected LS'])**2)/m)}
    resources = {'time_slots':t,'sensed_links':m,
        'pilot_energy_per_tx':np.sum(abs(p)**2,axis=0).tolist(),
        'communication_energy_per_tx':(pc*t).tolist(),
        'baseline_max_expected_slot_power':float(np.max(abs(p)**2+pc[None]))}
    return {'cee':cee,'nmse':nmse,'sic_ber':ber,'theory':theory,'resources':resources}
