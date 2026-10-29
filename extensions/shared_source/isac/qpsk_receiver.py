"""One exact finite-alphabet receiver for every transmit format.

Unknown users are marginalized over their QPSK alphabet, never replaced by
Gaussian interference for one method only. True bits enter scoring only.
"""
from functools import lru_cache
import numpy as np
from .models import cn
from .shared_source import source_packet


def logsumexp(a,axis):
    m=np.max(a,axis=axis,keepdims=True)
    return np.squeeze(m+np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True)),axis=axis)


@lru_cache(maxsize=32)
def alphabet(streams,desired):
    if not 1<=desired<=streams<=6:raise ValueError('Exact enumerator supports at most six QPSK streams.')
    labels=np.stack(np.unravel_index(np.arange(4**streams),(4,)*streams),axis=1)
    const=np.array([1+1j,1-1j,-1+1j,-1-1j])/np.sqrt(2)
    candidates=const[labels].T.copy()
    own=np.stack(np.unravel_index(np.arange(4**desired),(4,)*desired),axis=1)
    own_bits=np.stack((own//2,own%2),axis=-1).reshape(4**desired,2*desired)
    return candidates,own_bits


def posterior(y,a,desired_positions,sigma2,batch=48):
    """Return log p(desired QPSK vector | y,A), with no true data argument."""
    desired=list(desired_positions)
    perm=desired+[i for i in range(a.shape[2]) if i not in desired]
    aa=a[:,:,perm]
    q=len(perm);d=len(desired)
    candidates,own_bits=alphabet(q,d)
    result=np.empty((len(y),4**d))
    for start in range(0,len(y),batch):
        stop=min(start+batch,len(y))
        pred=(aa[start:stop].reshape(-1,q)@candidates).reshape(stop-start,a.shape[1],-1)
        scores=-np.sum(abs(y[start:stop,:,None]-pred)**2,axis=1)/sigma2
        grouped=logsumexp(scores.reshape(stop-start,4**d,4**(q-d)),axis=2)
        result[start:stop]=grouped-logsumexp(grouped,axis=1)[:,None]
    return result,own_bits


def score(logpost,own_bits,true_bits):
    """Information density and bit-MAP errors; ground truth is used only here."""
    desired=true_bits.shape[1]
    digits=2*true_bits[:,:,0]+true_bits[:,:,1]
    true_group=np.sum(digits*4**np.arange(desired-1,-1,-1)[None],axis=1)
    info=2*desired+logpost[np.arange(len(logpost)),true_group]/np.log(2)
    prob_one=np.exp(logpost)@own_bits
    detected=prob_one>.5
    errors=np.sum(detected!=true_bits.reshape(len(logpost),-1),axis=1)
    return info,errors


def communication(case,snr_db,trials,seed,batch=48):
    symbols,bits,rng_h,rng_n=source_packet(case,trials,seed)
    nr=3 if case.scenario=='BIA' else 5
    nt=case.encoded.sensing.shape[1];tmax=case.encoded.slots
    if case.scenario=='BIA':
        h=cn(rng_h,(trials,nr,tmax,nt))
    else:
        h=np.broadcast_to(cn(rng_h,(trials,nr,1,nt))*case.topology[None,:,None,:],(trials,nr,tmax,nt))
    sigma2=case.reference_power*10**(-snr_db/10)
    noise=np.sqrt(sigma2)*cn(rng_n,(trials,nr,tmax))
    output={};ranks=[]
    for fmt in (case.encoded,case.direct):
        t=fmt.slots
        hc=h[:,:,:t]
        data=np.einsum('tjs,ns->ntj',fmt.communication,symbols,optimize=True)
        sensing_contribution=np.einsum('nrtj,tj->nrt',hc,fmt.sensing,optimize=True)
        received=np.einsum('nrtj,ntj->nrt',hc,fmt.sensing[None]+data,optimize=True)+noise[:,:,:t]
        y=received-sensing_contribution
        residual=float(np.max(abs(y-(np.einsum('nrtj,ntj->nrt',hc,data,optimize=True)+noise[:,:,:t]))))
        info=np.zeros(trials);errors=np.zeros(trials,dtype=np.int64)
        for b in range(case.blocks):
            sl=slice(b*fmt.block,(b+1)*fmt.block)
            for r in range(nr):
                cb=fmt.communication[sl]*case.topology[r][None,:,None]
                active=np.flatnonzero(np.sum(abs(cb)**2,axis=(0,1))>1e-14)
                wanted=np.flatnonzero(case.owners[active]==r)
                a=np.einsum('ntj,tjs->nts',hc[:,r,sl],cb[:,:,active],optimize=True)
                logpost,labels=posterior(y[:,r,sl],a,wanted,sigma2,batch)
                density,count=score(logpost,labels,bits[:,active[wanted]])
                info+=density;errors+=count
                other=np.setdiff1d(np.arange(len(active)),wanted)
                a0=a[0]
                dim=int(np.linalg.matrix_rank(a0)-np.linalg.matrix_rank(a0[:,other]))
                ranks.append(dict(format=fmt.label,block=b,receiver=r+1,desired_streams=len(wanted),
                                  connected_streams=len(active),observations=fmt.block,
                                  desired_dimensions_after_interference_nulling=dim,zf_feasible=dim==len(wanted)))
        payload_bits=2*len(case.owners)
        output[fmt.label]=dict(ber=errors/payload_bits,bit_errors=errors,information_bits_per_frame=info,
                               rate_native=info/t,rate_common_window=info/tmax,payload_bits=payload_bits,
                               active_slots=t,common_window_slots=tmax,cancellation_error=residual)
    output['rank_audit']=ranks
    return output
