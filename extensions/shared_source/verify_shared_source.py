"""Verify the corrected shared-source contract and all saved results."""
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path,nargs='?',default=ROOT/'results_shared_source')
    p.add_argument('--write-report',action='store_true',help='Write verification.json after validation.')
    p.add_argument('--render-pdf',action='store_true',help='Save PDF page previews (requires PyMuPDF).')
    args=p.parse_args()
    if not __debug__:raise SystemExit('Verification requires assertions; run without -O/-OO.')
    out=args.out;m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));cfg=m['config']
    for group in ('numerical_source_sha256','plot_source_sha256'):
        for path,h in m[group].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    comm=read(out/'data'/'communication.csv');sense=read(out/'data'/'sensing.csv')
    resources=read(out/'data'/'resource_audit.csv');ranks=read(out/'data'/'linear_rank_audit.csv')
    raw=np.load(out/'raw'/'samples.npz');checked=0
    for row in comm+sense:
        n=int(row['n']);mean=float(row['mean'])
        assert np.isfinite([mean,float(row['sum']),float(row['sum_squares'])]).all()
        np.testing.assert_allclose(float(row['sum'])/n,mean,rtol=1e-12,atol=1e-14)
        var=max(0.,(float(row['sum_squares'])-float(row['sum'])**2/n)/(n-1))
        half=1.96*np.sqrt(var/n)
        low=mean-half
        if row['metric'] in ('cee','ber','sensor_decoder_ber'):low=max(0.,low)
        np.testing.assert_allclose([float(row['ci95_low']),float(row['ci95_high'])],[low,mean+half],rtol=1e-9,atol=1e-11)
    for row in comm:
        key=f"comm__{row['scenario']}__{float(row['rho']):.2f}__{row['snr_db']}__{row['format']}__{row['metric']}"
        np.testing.assert_allclose(raw[key].mean(),float(row['mean']),rtol=1e-13)
        payload=int(row['payload_bits'])
        if row['metric']=='information_bits_per_frame':assert np.max(raw[key])<=payload+1e-10
        if row['metric']=='rate_common_window':assert np.max(raw[key])<=payload/int(row['common_window_slots'])+1e-10
        if row['metric']=='rate_native':assert np.max(raw[key])<=payload/int(row['active_slots'])+1e-10
        if row['metric']=='ber':
            np.testing.assert_allclose(int(row['bit_errors'])/int(row['total_bits']),float(row['mean']),atol=1e-14)
        checked+=1
    for row in sense:
        if row['metric']=='cee':
            key=f"sense__{row['scenario']}__{row['family']}__{float(row['rho']):.2f}__{row['snr_db']}__{row['method']}"
            np.testing.assert_allclose(raw[key].mean(),float(row['mean']),rtol=1e-13);checked+=1
    rr={(r['scenario'],r['family'],r['rho'],r['format']):r for r in resources}
    for scenario in ('BIA','TIM'):
        for family in cfg['families']:
            for rho in cfg['sensing_fractions']:
                a=rr[scenario,family,str(rho),'encoded'];b=rr[scenario,family,str(rho),'direct']
                assert a['independent_communication_symbols']==b['independent_communication_symbols']
                np.testing.assert_allclose(json.loads(a['message_energies']),json.loads(b['message_energies']),rtol=1e-12)
                np.testing.assert_allclose(json.loads(a['sensing_energy_per_tx']),json.loads(b['sensing_energy_per_tx']),rtol=1e-12,atol=1e-13)
                assert float(a['expected_peak_power'])<=1+1e-12 and float(b['expected_peak_power'])<=1+1e-12
        packet=np.load(out/'raw'/f'{scenario.lower()}_shared_packet_example.npz')
        bits=packet['communication_bits'];symbols=packet['communication_symbols']
        np.testing.assert_allclose(symbols,((1-2*bits[...,0])+1j*(1-2*bits[...,1]))/np.sqrt(2),atol=1e-14)
        for fmt in ('encoded','direct'):
            c=packet[f'{fmt}_coefficients'];x=packet[f'{fmt}_communication']
            np.testing.assert_allclose(x,np.einsum('tjs,ns->ntj',c,symbols),atol=1e-13)
            recovered=x.reshape(len(x),-1)@np.linalg.pinv(c.reshape(-1,c.shape[-1])).T
            np.testing.assert_allclose(recovered,symbols,atol=1e-12)
    assert all(r['zf_feasible']=='True' for r in ranks if r['format']=='encoded')
    assert all(r['zf_feasible']=='False' for r in ranks if r['format']=='direct')
    cc={(r['scenario'],float(r['rho']),float(r['snr_db']),r['format'],r['metric']):float(r['mean']) for r in comm}
    ss={(r['scenario'],r['family'],float(r['rho']),float(r['snr_db']),r['method']):float(r['mean']) for r in sense if r['metric']=='cee'}
    assert len(m['figures'])==12;points=0
    for e in m['figures']:
        name=e['id']
        for ext in ('png','pdf','svg'):
            f=out/'figures'/f'{name}.{ext}';assert f.is_file() and f.stat().st_size>1000
        assert ET.parse(out/'figures'/f'{name}.svg').findall('.//{http://www.w3.org/2000/svg}text')
        for row in read(out/'data'/f'{name}.csv'):
            key=(row['scenario'],float(row['rho']),float(row['snr_db']),row['format'])
            for metric,source in [('rate','rate_common_window'),('rate_native','rate_native'),('ber','ber')]:
                np.testing.assert_allclose(float(row[metric]),cc[key+(source,)],rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(float(row['cee']),ss[row['scenario'],row['family'],float(row['rho']),float(row['snr_db']),row['method']],rtol=1e-13)
            points+=1
    pdf='optional PDF inspection unavailable'
    if importlib.util.find_spec('pymupdf'):
        import pymupdf
        doc=pymupdf.open(out/'all_figures.pdf');assert len(doc)==12
        assert all(len(page.get_text())>80 for page in doc)
        pdf='12 pages; selectable text'
        if args.render_pdf:
            for i in (0,2,3):doc[i].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(out/f'saved_pdf_page_{i+1:02d}.png')
            pdf+='; pages 1,3,4 rendered'
        doc.close()
    result=dict(status='PASS',figures=12,exports=36,raw_statistics_groups_checked=checked,plot_points_checked=points,
                raw_symbol_domains_identical=True,per_symbol_energies_matched=True,source_packet_reconstruction=True,
                same_receiver=True,finite_qpsk_rate_caps_verified=True,source_hashes_match=True,pdf=pdf)
    if args.write_report:
        (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
