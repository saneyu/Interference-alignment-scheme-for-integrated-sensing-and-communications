"""Shared raw sources, direct-superposition baseline, identical QPSK MAP receiver."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from isac.shared_source import make_case,shared_power,sensing,source_packet
from isac.qpsk_receiver import communication

ROOT=Path(__file__).resolve().parent


def stats(a,nonnegative=True):
    a=np.asarray(a,float);mean=float(a.mean());half=1.96*float(a.std(ddof=1))/np.sqrt(len(a))
    return dict(n=len(a),mean=mean,ci95_low=max(0.,mean-half) if nonnegative else mean-half,
                ci95_high=mean+half,sum=float(a.sum()),sum_squares=float(np.dot(a,a)),
                zero_error_upper_95=(1-.05**(1/len(a)) if np.all(a==0) else ''))


def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/shared_source.json')
    p.add_argument('--out',type=Path,default=ROOT/'results_shared_source')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--simulate-only',action='store_true')
    args=p.parse_args();out=args.out
    cfg=json.loads(args.config.read_text())
    if args.quick:cfg.update(communication_trials=100,sensing_trials=400)
    for d in ('data','raw','figures'):(out/d).mkdir(parents=True,exist_ok=True)
    power=shared_power(cfg['families'],cfg['sensing_fractions'],cfg['blocks'],cfg['waveform_seed'],cfg['peak_expected_power_cap'])
    cfg['reference_power']=power
    (out/'config_used.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    started=time.perf_counter();comm_rows=[];sense_rows=[];ranks=[];resources=[];samples={}
    max_cancel=0.;max_peak=0.;theory_deviations=[]
    for scenario in ('BIA','TIM'):
        for rho in cfg['sensing_fractions']:
            case=make_case(scenario,'dft',rho,power,cfg['blocks'],cfg['waveform_seed'])
            for snr in cfg['snr_db']:
                print(f'Common QPSK MAP: {scenario}, rho={rho}, SNR={snr}',flush=True)
                result=communication(case,snr,cfg['communication_trials'],cfg['seed']+1000,cfg['likelihood_batch'])
                if rho==cfg['main_sensing_fraction'] and snr==cfg['snr_db'][0]:
                    ranks.extend(dict(scenario=scenario,**r) for r in result['rank_audit'])
                for fmt in ('encoded','direct'):
                    v=result[fmt];max_cancel=max(max_cancel,v['cancellation_error'])
                    assert np.max(v['information_bits_per_frame'])<=v['payload_bits']+1e-10
                    for metric in ('ber','information_bits_per_frame','rate_native','rate_common_window'):
                        row=dict(scenario=scenario,rho=rho,snr_db=snr,format=fmt,metric=metric,
                                 active_slots=v['active_slots'],common_window_slots=v['common_window_slots'],
                                 payload_bits=v['payload_bits'],**stats(v[metric],nonnegative=metric=='ber'))
                        if metric=='ber':row.update(bit_errors=int(v['bit_errors'].sum()),total_bits=v['payload_bits']*cfg['communication_trials'])
                        comm_rows.append(row)
                        samples[f'comm__{scenario}__{rho:.2f}__{snr}__{fmt}__{metric}']=v[metric]
        # Save one explicit input/transmit example for an auditable shared payload.
        c=make_case(scenario,'dft',cfg['main_sensing_fraction'],power,cfg['blocks'],cfg['waveform_seed'])
        raw,bits,_,_=source_packet(c,1,cfg['seed'])
        np.savez_compressed(out/'raw'/f'{scenario.lower()}_shared_packet_example.npz',
                            communication_symbols=raw,communication_bits=bits,raw_sensing=c.raw_sensing,
                            encoded_sensing=c.encoded.sensing,direct_sensing=c.direct.sensing,
                            encoded_communication=np.einsum('tjs,ns->ntj',c.encoded.communication,raw),
                            direct_communication=np.einsum('tjs,ns->ntj',c.direct.communication,raw),
                            encoded_coefficients=c.encoded.communication,direct_coefficients=c.direct.communication)
    csvwrite(out/'data'/'communication.csv',comm_rows);csvwrite(out/'data'/'linear_rank_audit.csv',ranks)
    for scenario in ('BIA','TIM'):
        for family in cfg['families']:
            for rho in cfg['sensing_fractions']:
                print(f'Shared-source sensing: {scenario}/{family}, rho={rho}',flush=True)
                case=make_case(scenario,family,rho,power,cfg['blocks'],cfg['waveform_seed'])
                encoded=case.encoded;direct=case.direct
                for fmt in (encoded,direct):
                    peak=float(np.max(abs(fmt.sensing)**2+np.sum(abs(fmt.communication)**2,axis=2)))
                    max_peak=max(max_peak,peak)
                    resources.append(dict(scenario=scenario,family=family,rho=rho,format=fmt.label,
                        active_slots=fmt.slots,common_window_slots=encoded.slots,
                        independent_communication_symbols=len(case.owners),raw_sensing_columns=len(case.raw_sensing),
                        message_energies=np.sum(abs(fmt.communication)**2,axis=(0,1)).tolist(),
                        sensing_energy_per_tx=np.sum(abs(fmt.sensing)**2,axis=0).tolist(),
                        expected_peak_power=peak))
                np.testing.assert_allclose(np.sum(abs(encoded.communication)**2,axis=(0,1)),
                                           np.sum(abs(direct.communication)**2,axis=(0,1)),rtol=1e-12)
                np.testing.assert_allclose(np.sum(abs(encoded.sensing)**2,axis=0),
                                           np.sum(abs(direct.sensing)**2,axis=0),rtol=1e-12,atol=1e-13)
                for snr in cfg['snr_db']:
                    errors,theory,ber=sensing(case,snr,cfg['sensing_trials'],cfg['seed'],cfg['sic_iterations'])
                    for method in (scenario,'SIC','TIN'):
                        row=dict(scenario=scenario,family=family,rho=rho,snr_db=snr,method=method,metric='cee',
                                 analytic_cee=theory.get(method,''),**stats(errors[method]))
                        sense_rows.append(row)
                        if method in theory:theory_deviations.append(abs(row['mean']/theory[method]-1))
                        samples[f'sense__{scenario}__{family}__{rho:.2f}__{snr}__{method}']=errors[method]
                    sense_rows.append(dict(scenario=scenario,family=family,rho=rho,snr_db=snr,method='SIC',metric='sensor_decoder_ber',**stats(ber)))
    csvwrite(out/'data'/'sensing.csv',sense_rows);csvwrite(out/'data'/'resource_audit.csv',resources)
    np.savez_compressed(out/'raw'/'samples.npz',**samples)
    assert max_peak<=cfg['peak_expected_power_cap']+1e-12 and max_cancel<1e-12
    sources=['run_shared_source.py','isac/shared_source.py','isac/qpsk_receiver.py','isac/models.py','isac/fixed_waveforms.py','tests/test_shared_source.py']
    manifest=dict(config=cfg,simulation_seconds=time.perf_counter()-started,
                  exact_same_raw_source=True,same_communication_receiver=True,receiver='QPSK marginal bit-MAP',
                  extra_baseline_independent_symbols=0,peak_power_max=max_peak,max_known_sensing_cancellation_error=max_cancel,
                  max_sensing_theory_relative_deviation=max(theory_deviations),
                  numerical_source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('simulation_seconds','exact_same_raw_source','same_communication_receiver','peak_power_max','max_sensing_theory_relative_deviation')},indent=2),flush=True)
    if not args.simulate_only:
        from isac.shared_source_plots import make_figures
        make_figures(out)


if __name__=='__main__':main()
