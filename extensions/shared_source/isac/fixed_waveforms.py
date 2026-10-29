"""Fixed sensing sequences through the original BIA/TIM encoding maps.

No optimizer, channel-conditioned sequence, seed search, or orthogonalization.
Power fractions are actual time-averaged component energies, with shared backoff.
"""
from dataclasses import replace
import numpy as np
from .models import bia, tim, cn, qpsk
from .benchmarks import sic_estimate

LABELS = {'dft':'DFT tones', 'lfm':'Fixed LFM', 'walsh':'Walsh / BPSK', 'qpsk':'Fixed QPSK'}


def sequence(family, length, channels=3, seed=314159):
    if length < channels:
        raise ValueError('Sequence too short for the channel dimension.')
    n=np.arange(length)
    tones=np.exp(-2j*np.pi*np.outer(n,np.arange(channels))/length)
    if family=='dft':
        s=tones
    elif family=='lfm':
        # Fixed quadratic phase (sampled linear frequency sweep), fixed offsets.
        s=tones*np.exp(1j*np.pi*(n-(length-1)/2)**2/length)[:,None]
    elif family=='walsh':
        h=np.ones((1,1))
        while len(h)<length:
            h=np.block([[h,h],[h,-h]])
        s=h[:channels,:length].T.astype(complex)
    elif family=='qpsk':
        s,_=qpsk(np.random.default_rng(seed+length),(length,channels))
    else:
        raise ValueError(f'Unknown fixed sequence {family}.')
    if np.linalg.matrix_rank(s)!=channels:
        raise ValueError(f'{family}, length={length}, seed={seed} is rank deficient. No resampling performed.')
    return s


def unit_mean_columns(s):
    energy=np.mean(abs(s)**2,axis=0)
    active=energy>1e-14
    out=s.copy()
    out[:,active]/=np.sqrt(energy[active])[None,:]
    return out


def shapes(scenario,family,blocks=2,seed=314159):
    if scenario=='BIA':
        original=bia(3,blocks,.5,lift='paper')
        a=np.ones((3,2));a[1:]-=np.eye(2)
        source=sequence(family,blocks*2,seed=seed)
        sensing=np.vstack([a@source[2*b:2*b+2] for b in range(blocks)])
    elif scenario=='TIM':
        original=tim(blocks,.5)
        u=original.q[:5,:2]
        source=sequence(family,blocks*2,seed=seed)
        sensing=np.zeros_like(original.pilot)
        for b in range(blocks):
            sensing[b*5:(b+1)*5,original.sensed]=u@source[2*b:2*b+2]
    else:
        raise ValueError('Scenario must be BIA or TIM.')
    sensing=unit_mean_columns(sensing)
    baseline=unit_mean_columns(sequence(family,original.T,seed=seed))
    if np.linalg.matrix_rank(original.q.conj().T@sensing[:,original.sensed])!=3:
        raise ValueError('Encoded sensing waveform is not identifiable.')
    return original,sensing,baseline


def common_average_power(families,fractions,blocks=2,seed=314159,peak_cap=1.):
    factor=1.
    for scenario in ('BIA','TIM'):
        for family in families:
            original,sensing,baseline=shapes(scenario,family,blocks,seed)
            comm_unit=original.comm/np.sqrt(.5)
            comm_power=np.sum(abs(comm_unit)**2,axis=2)
            ids=original.sensed
            pc=np.mean(comm_power[:,ids],axis=0)
            for rho in fractions:
                factor=max(factor,float(np.max(rho*abs(sensing)**2+(1-rho)*comm_power)),
                           float(np.max(rho*abs(baseline)**2+(1-rho)*pc[None])))
    return float(peak_cap/factor)


def build(scenario,family,rho,power,blocks=2,seed=314159):
    if not 0<rho<1 or power<=0:
        raise ValueError('Need 0<rho<1 and positive power.')
    original,sensing,baseline=shapes(scenario,family,blocks,seed)
    sensing=sensing*np.sqrt(rho*power)
    comm=original.comm*np.sqrt((1-rho)*power/.5)
    design=replace(original,pilot=sensing,comm=comm,alpha=rho)
    baseline=baseline*np.sqrt(rho*power)
    return design,baseline


def diagnostics(design,baseline,power):
    ids=design.sensed
    a=design.q.conj().T@design.pilot[:,ids]
    pc=np.mean(np.sum(abs(design.comm[:,ids])**2,axis=2),axis=0)
    return dict(
        sensing_energy=np.sum(abs(design.pilot[:,ids])**2,axis=0).tolist(),
        baseline_sensing_energy=np.sum(abs(baseline)**2,axis=0).tolist(),
        communication_energy=(pc*design.T).tolist(),
        expected_power_peak=float(design.expected_power.max()),
        baseline_expected_power_peak=float(np.max(abs(baseline)**2+pc[None])),
        actual_sensing_fraction=float(design.alpha),
        common_average_power=power,
        projected_condition=float(np.linalg.cond(a)),
        baseline_condition=float(np.linalg.cond(baseline)),
        sensing_papr_db=(10*np.log10(np.max(abs(design.pilot[:,ids])**2,axis=0)/
                                     np.mean(abs(design.pilot[:,ids])**2,axis=0))).tolist(),
        nulling_residual=float(np.linalg.norm(design.q.conj().T@design.comm[:,ids].reshape(design.T,-1))),
        time_slots=design.T
    )


def simulate(design,baseline,snr_db,trials,seed,power,iterations=4):
    rng_h,rng_n,rng_a,rng_b=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    ids=design.sensed;m=len(ids)
    h=cn(rng_h,(trials,m))
    sigma2=power*10**(-snr_db/10)
    noise=np.sqrt(sigma2)*cn(rng_n,(trials,design.T))
    pc=np.mean(np.sum(abs(design.comm[:,ids])**2,axis=2),axis=0)
    s_base,bits=qpsk(rng_b,(trials,design.T,m))
    x_base=s_base*np.sqrt(pc)[None,None]
    y_base=np.einsum('ntm,nm->nt',baseline[None]+x_base,h)+noise
    h_tin=y_base@np.linalg.pinv(baseline).T
    h_sic,decoded=sic_estimate(y_base,baseline,pc,iterations)
    symbols,_=qpsk(rng_a,(trials,design.comm.shape[2]))
    x_aligned=np.einsum('tms,ns->ntm',design.comm[:,ids],symbols,optimize=True)
    y=np.einsum('ntm,nm->nt',design.pilot[:,ids][None]+x_aligned,h)+noise
    op=np.linalg.pinv(design.q.conj().T@design.pilot[:,ids])@design.q.conj().T
    h_prop=y@op.T
    norm=np.sum(abs(h)**2,axis=1)
    cee={method:np.sum(abs(hh-h)**2,axis=1)/norm
         for method,hh in ((design.name,h_prop),('SIC',h_sic),('TIN',h_tin))}
    active=np.flatnonzero(pc>1e-14)
    predicted=np.stack((decoded.real<0,decoded.imag<0),axis=-1)
    sic_ber=np.mean(predicted[:,:,active]!=bits[:,:,active],axis=(1,2,3))
    noise_gain=np.sum(abs(np.linalg.pinv(baseline))**2)
    theory={design.name:float(sigma2*np.sum(abs(op)**2)/(m-1)),
            'TIN':float(noise_gain*(sigma2/(m-1)+pc.sum()/m))}
    return cee,theory,sic_ber
