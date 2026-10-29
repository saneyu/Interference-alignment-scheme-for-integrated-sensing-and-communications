"""Identical raw symbols; original encoding versus literal direct superposition."""
from dataclasses import dataclass
import numpy as np
from .models import bia,tim,cn,qpsk
from .fixed_waveforms import sequence


@dataclass
class Format:
    label: str
    block: int
    sensing: np.ndarray
    communication: np.ndarray

    @property
    def slots(self):return self.sensing.shape[0]


@dataclass
class Case:
    scenario: str
    blocks: int
    raw_sensing: np.ndarray
    owners: np.ndarray
    sensed: np.ndarray
    topology: np.ndarray
    projector: np.ndarray
    encoded: Format
    direct: Format
    rho: float
    reference_power: float


def normalize_energy(s,c,sensing_energy,message_energy):
    out=s.copy()
    col_energy=np.sum(abs(s)**2,axis=0)
    active=col_energy>1e-14
    out[:,active]*=np.sqrt(sensing_energy/col_energy[active])[None]
    message_norm=np.sum(abs(c)**2,axis=(0,1))
    if np.any(message_norm<1e-14):raise ValueError('A source message is missing.')
    return out,c*np.sqrt(message_energy/message_norm)[None,None]


def make_case(scenario,family,rho,power=1.,blocks=2,seed=314159):
    original=bia(3,blocks,.5) if scenario=='BIA' else tim(blocks,.5)
    nt=original.n_tx;nr=3 if scenario=='BIA' else 5
    raw=sequence(family,2*blocks,seed=seed)
    raw_direct=np.zeros((2*blocks,nt),complex)
    raw_direct[:,original.sensed]=raw
    raw_encoded=np.zeros_like(original.pilot)
    if scenario=='BIA':
        lift=np.array([[1,1],[0,1],[1,0]],complex)
        for b in range(blocks):raw_encoded[3*b:3*b+3]=lift@raw[2*b:2*b+2]
        count=blocks*nr
        owners=np.tile(np.arange(nr),blocks)
        c_direct=np.zeros((2*blocks,nt,count),complex)
        for b in range(blocks):
            for j in range(nr):c_direct[2*b:2*b+2,j,b*nr+j]=1
        messages_per_tx=blocks
        top=np.ones((nr,nt),bool)
    else:
        lift=original.q[:5,:2]
        for b in range(blocks):
            raw_encoded[b*5:b*5+5,original.sensed]=lift@raw[2*b:2*b+2]
        count=blocks*nr*2
        owners=np.tile(np.repeat(np.arange(nr),2),blocks)
        c_direct=np.zeros((2*blocks,nt,count),complex)
        for b in range(blocks):
            for j in range(nr):
                for u in range(2):c_direct[2*b+u,j,b*nr*2+j*2+u]=1
        messages_per_tx=blocks*2
        top=original.topology[:nr]
    es=rho*power*original.T
    ec=(1-rho)*power*original.T/messages_per_tx
    s_enc,c_enc=normalize_energy(raw_encoded,original.comm,es,ec)
    s_dir,c_dir=normalize_energy(raw_direct,c_direct,es,ec)
    return Case(scenario,blocks,raw.copy(),owners,original.sensed,top,original.q,
                Format('encoded',original.block,s_enc,c_enc),
                Format('direct',2,s_dir,c_dir),rho,power)


def shared_power(families,rhos,blocks=2,seed=314159,peak_cap=1.):
    largest=1.
    for scenario in ('BIA','TIM'):
        for family in families:
            for rho in rhos:
                case=make_case(scenario,family,rho,1.,blocks,seed)
                for f in (case.encoded,case.direct):
                    largest=max(largest,float(np.max(abs(f.sensing)**2+np.sum(abs(f.communication)**2,axis=2))))
    return peak_cap/largest


def source_packet(case,trials,seed):
    """Only one random communication source exists; both transmitters use this array."""
    children=np.random.SeedSequence(seed).spawn(3)
    rng_data,rng_h,rng_noise=[np.random.default_rng(s) for s in children]
    symbols,bits=qpsk(rng_data,(trials,len(case.owners)))
    return symbols,bits,rng_h,rng_noise


def received_sensing(case,snr_db,trials,seed):
    symbols,bits,rng_h,rng_n=source_packet(case,trials,seed)
    h=cn(rng_h,(trials,len(case.sensed)))
    sigma2=case.reference_power*10**(-snr_db/10)
    noise=np.sqrt(sigma2)*cn(rng_n,(trials,case.encoded.slots))
    observations={};transmitted={}
    for fmt in (case.encoded,case.direct):
        x=fmt.sensing[None]+np.einsum('tjs,ns->ntj',fmt.communication,symbols,optimize=True)
        observations[fmt.label]=np.einsum('ntj,nj->nt',x[:,:,case.sensed],h)+noise[:,:fmt.slots]
        transmitted[fmt.label]=x
    return h,observations,symbols,bits,sigma2


def source_aware_sic(y,sensing,c,iterations=4):
    """SIC of the actual shared raw symbols, respecting their temporal signatures."""
    n=y.shape[0]
    hhat=y@np.linalg.pinv(sensing).T
    active=np.flatnonzero(np.sum(abs(c)**2,axis=(0,1))>1e-14)
    decisions=np.zeros((n,c.shape[2]),complex)
    qconst=np.array([1+1j,1-1j,-1+1j,-1-1j])/np.sqrt(2)
    for _ in range(iterations):
        v=np.einsum('tjs,nj->nts',c[:,:,active],hhat,optimize=True)
        energies=np.sum(abs(v)**2,axis=1)
        order=np.argsort(-energies,axis=1)
        residual=y-hhat@sensing.T
        for stage in range(len(active)):
            index=order[:,stage]
            signature=v[np.arange(n),:,index]
            soft=np.sum(signature.conj()*residual,axis=1)/np.maximum(energies[np.arange(n),index],1e-30)
            decoded=qconst[2*(soft.real<0)+(soft.imag<0)]
            decisions[np.arange(n),active[index]]=decoded
            residual-=signature*decoded[:,None]
        x=sensing[None]+np.einsum('tjs,ns->ntj',c,decisions,optimize=True)
        hhat=np.einsum('nmt,nt->nm',np.linalg.pinv(x),y)
    return hhat,decisions


def sensing(case,snr_db,trials,seed,iterations=4):
    h,y,symbols,bits,sigma2=received_sensing(case,snr_db,trials,seed)
    ids=case.sensed
    a=case.projector.conj().T@case.encoded.sensing[:,ids]
    ep=np.linalg.pinv(a)@case.projector.conj().T
    sd=case.direct.sensing[:,ids]
    ed=np.linalg.pinv(sd)
    cd=case.direct.communication[:,ids]
    hs,decisions=source_aware_sic(y['direct'],sd,cd,iterations)
    estimates={case.scenario:y['encoded']@ep.T,'TIN':y['direct']@ed.T,'SIC':hs}
    norm=np.sum(abs(h)**2,axis=1)
    errors={m:np.sum(abs(hh-h)**2,axis=1)/norm for m,hh in estimates.items()}
    ri=np.einsum('tjs,ujs->tu',cd,cd.conj())
    theory={case.scenario:float(sigma2*np.sum(abs(ep)**2)/(len(ids)-1)),
            'TIN':float(sigma2*np.sum(abs(ed)**2)/(len(ids)-1)+np.trace(ed@ri@ed.conj().T).real/len(ids))}
    active=np.flatnonzero(np.sum(abs(cd)**2,axis=(0,1))>1e-14)
    decoded_bits=np.stack((decisions.real<0,decisions.imag<0),axis=-1)
    ber=np.mean(decoded_bits[:,active]!=bits[:,active],axis=(1,2))
    return errors,theory,ber
