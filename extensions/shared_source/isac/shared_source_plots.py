"""Shared-source results, explicitly distinguishing duration and rate definitions."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from .fixed_waveforms import LABELS
from .plotting import style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COLORS=['#52788D','#668C7A','#BC8450','#8B83A3']
LABEL={'rate':'QPSK information rate, common window\n(bit/channel use)',
       'rate_native':'QPSK information rate, actual airtime\n(bit/channel use)',
       'cee':'Mean sensing CEE (dB)','ber':'Communication QPSK BER',
       'snr_db':'Reference SNR (dB)','rho':'Sensing energy fraction'}


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


class Grid:
    def __init__(self,out):
        self.cfg=json.loads((out/'config_used.json').read_text(encoding='utf-8'))
        self.c={(r['scenario'],float(r['rho']),float(r['snr_db']),r['format'],r['metric']):r
                for r in read(out/'data'/'communication.csv')}
        self.s={(r['scenario'],r['family'],float(r['rho']),float(r['snr_db']),r['method']):r
                for r in read(out/'data'/'sensing.csv') if r['metric']=='cee'}

    def point(self,scenario,family,rho,snr,method):
        fmt='encoded' if method==scenario else 'direct'
        c=self.c[scenario,rho,snr,fmt,'rate_common_window'];n=self.c[scenario,rho,snr,fmt,'rate_native']
        b=self.c[scenario,rho,snr,fmt,'ber'];s=self.s[scenario,family,rho,snr,method]
        zero=float(b['mean'])==0
        return dict(scenario=scenario,family=family,method=method,format=fmt,rho=rho,snr_db=snr,
                    rate=float(c['mean']),rate_native=float(n['mean']),ber=float(b['mean']),cee=float(s['mean']),
                    display_ber=float(b['zero_error_upper_95']) if zero else float(b['mean']),
                    ber_is_upper_limit=zero,bit_errors=int(b['bit_errors']),total_bits=int(b['total_bits']),
                    active_slots=int(c['active_slots']),common_window_slots=int(c['common_window_slots']),
                    payload_bits=int(c['payload_bits']),communication_trials=int(c['n']),sensing_trials=int(s['n']),
                    rate_ci95_low=float(c['ci95_low']),rate_ci95_high=float(c['ci95_high']),
                    cee_ci95_low=float(s['ci95_low']),cee_ci95_high=float(s['ci95_high']))

    def series(self,scenario,family,method,var,values,fixed):
        return [self.point(scenario,family,v if var=='rho' else fixed,fixed if var=='rho' else v,method) for v in values]


class Figures:
    def __init__(self,out):
        self.out=out;self.pdf=PdfPages(out/'all_figures.pdf');self.entries=[];self.rows=[]
    def new(self,rows,cols,size,title):
        self.rows=[];fig,axes=plt.subplots(rows,cols,figsize=size,layout='constrained',squeeze=False)
        fig.suptitle(title,fontsize=11);return fig,axes
    def line(self,ax,points,x,y,label,i):
        xx=np.array([p['display_ber'] if x=='ber' else p[x] for p in points])
        yy=np.array([p['display_ber'] if y=='ber' else p[y] for p in points])
        if y=='cee':yy=10*np.log10(yy)
        yy_line=yy.copy()
        if y=='ber':yy_line[np.array([p['ber_is_upper_limit'] for p in points])]=np.nan
        ax.plot(xx,yy_line,label=label,color=COLORS[i%4],marker=['o','s','^','D'][i%4],
                linestyle=['-','--','-.',':'][i%4],markersize=3.5,linewidth=1.4)
        if y=='ber':
            ax.set_yscale('log')
            zero=np.array([p['ber_is_upper_limit'] for p in points])
            if zero.any():ax.scatter(xx[zero],yy[zero],marker='v',s=40,facecolors='white',edgecolors=COLORS[i%4],zorder=5,label='Zero errors: 95% upper limit' if i==0 else None)
        ax.set(xlabel=LABEL[x],ylabel=LABEL[y])
        for p,xv,yv in zip(points,xx,yy):
            self.rows.append(dict(**p,curve_label=label,x_metric=x,y_metric=y,plotted_x=float(xv),plotted_y=float(yv)))
    def save(self,name,fig,caption):
        for ext in ('png','pdf','svg'):fig.savefig(self.out/'figures'/f'{name}.{ext}',bbox_inches='tight')
        self.pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        keys=list(dict.fromkeys(k for row in self.rows for k in row))
        with (self.out/'data'/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(self.rows)
        self.entries.append(dict(id=name,caption=caption))
    def finish(self):
        self.pdf.close()
        fig,axes=plt.subplots(4,3,figsize=(17,17),layout='constrained')
        for ax,e in zip(axes.flat,self.entries):
            ax.imshow(plt.imread(self.out/'figures'/f'{e["id"]}.png'));ax.axis('off');ax.set_title(e['id'],fontsize=8)
        for ax in axes.flat[len(self.entries):]:ax.axis('off')
        fig.savefig(self.out/'overview.png',dpi=150);plt.close(fig)
        html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>共享原始符号对比</title>',
              '<style>body{font:16px system-ui;max-width:1150px;margin:36px auto;color:#243746}article{background:#fafafa;padding:22px;margin:24px 0}img{width:100%}a{color:#52788d}</style>',
              '<h1>同原始符号、直接叠加、统一 QPSK 接收机</h1><p>各方法原始符号相同，逐通信符号能量和感知总能量匹配。接收规则完全一致；实际时长与共同窗口分别归一化。没有按曲线优势挑选场景。</p><a href="all_figures.pdf">全部图 PDF</a>']
        for e in self.entries:
            s=e['id'];html.append(f'<article><h2>{s}</h2><img src="figures/{s}.png"><p>{e["caption"]}</p><a href="figures/{s}.svg">SVG</a> · <a href="data/{s}.csv">CSV</a></article>')
        html.append('</html>');(self.out/'index.html').write_text('\n'.join(html),encoding='utf-8')


def make_figures(out):
    out=Path(out);g=Grid(out);cfg=g.cfg;style();book=Figures(out)
    snrs=cfg['snr_db'];rhos=cfg['sensing_fractions'];main=cfg['main_sensing_fraction'];families=cfg['families']
    for number,metric in ((1,'rate'),(2,'ber')):
        fig,axes=book.new(1,2,(11,4.7),f'Identical QPSK MAP receiver and raw payload; rho={main:g}')
        for ax,scenario in zip(axes.flat,('BIA','TIM')):
            for i,method in enumerate((scenario,'TIN')):
                label=scenario if i==0 else 'Direct superposition (sensor SIC/TIN)'
                book.line(ax,g.series(scenario,'dft',method,'snr_db',snrs,main),'snr_db',metric,label,i)
            if metric=='rate':
                cap=2 if scenario=='BIA' else 4
                ax.axhline(cap,color='#777777',linestyle=':',linewidth=1,label=f'Payload ceiling = {cap}')
                ax.set_ylim(0,cap*1.08)
            ax.set_title(f'{scenario} scenario');ax.legend(fontsize=7.5)
        book.save(f'{number:02d}_common_receiver_{metric}',fig,
            '通信使用完全相同的 QPSK 边缘后验/逐比特 MAP 算法，无真实数据辅助判决。共同窗口下每帧原始比特数相同。零误码点用独立帧事件的保守 95% 上限显示，不当作 BER 精确为零。')
    fig,axes=book.new(2,2,(11,8),'Rate normalization: common deadline versus actual airtime')
    for row,scenario in enumerate(('BIA','TIM')):
        for ax,metric in zip(axes[row],('rate','rate_native')):
            for i,method in enumerate((scenario,'TIN')):
                book.line(ax,g.series(scenario,'dft',method,'snr_db',snrs,main),'snr_db',metric,
                          scenario if i==0 else 'Direct superposition',i)
            ax.set_title(f'{scenario}: '+('common window' if metric=='rate' else 'actual duration'));ax.legend(fontsize=8)
    book.save('03_duration_normalization',fig,
        '两边共享原始数据和帧能量，但编码有时间扩展。共同窗口分别为 6/10 时隙，直接叠加实际为 4 时隙。此图显示这项开销，避免把基线的空闲时间隐去或无端计为劣势。')
    for number,family in enumerate(families,4):
        fig,axes=book.new(2,2,(11,8),f'{LABELS[family]}: shared-source power tradeoffs, common window')
        for row,scenario in enumerate(('BIA','TIM')):
            for ax,snr in zip(axes[row],(10,30)):
                for i,method in enumerate((scenario,'SIC','TIN')):
                    book.line(ax,g.series(scenario,family,method,'rho',rhos,snr),'rate','cee',method,i)
                cap=2 if scenario=='BIA' else 4
                ax.set_xlim(0,cap*1.04);ax.set_title(f'{scenario}, SNR = {snr} dB');ax.legend(fontsize=8)
        book.save(f'{number:02d}_{family}_shared_source_tradeoff',fig,
            '同一份原始感知/通信符号，原编码与直接叠加不同。横轴为共同窗口 QPSK 互信息，纵轴为平均 CEE；沿曲线改变 rho。没有给基线增加新数据或弱化通信检测器。')
    fig,axes=book.new(1,2,(11,4.7),f'Fixed sensing sequences under original encoding; rho={main:g}')
    for ax,scenario in zip(axes.flat,('BIA','TIM')):
        for i,family in enumerate(families):
            book.line(ax,g.series(scenario,family,scenario,'snr_db',snrs,main),'snr_db','cee',LABELS[family],i)
        ax.set_title(scenario);ax.legend(fontsize=8)
    book.save('08_fixed_sensing_sequences',fig,'只比较固定基础序列，不进行专门波形优化。通信接收机已知感知分量，四类感知序列不会因此产生不同通信数据源。')
    for number,scenario in enumerate(('BIA','TIM'),9):
        fig,axes=book.new(2,2,(11,8),f'{scenario}: sensing with shared raw symbols; rho={main:g}')
        for ax,family in zip(axes.flat,families):
            for i,method in enumerate((scenario,'SIC','TIN')):
                book.line(ax,g.series(scenario,family,method,'snr_db',snrs,main),'snr_db','cee',method,i)
            ax.set_title(LABELS[family]);ax.legend(fontsize=8)
        book.save(f'{number:02d}_{scenario.lower()}_shared_source_sensing',fig,
            '感知端的差别来自发送映射及处理方法，不是换了原始数据。SIC 依照真实原始通信符号的时间签名检测，不再给每个观测时隙独立假设一个新符号。')
    fig,axes=book.new(2,3,(14,8),'Shared-source tradeoffs at several SNRs, fixed DFT sequence')
    for row,scenario in enumerate(('BIA','TIM')):
        for ax,snr in zip(axes[row],(0,20,35)):
            for i,method in enumerate((scenario,'SIC','TIN')):
                book.line(ax,g.series(scenario,'dft',method,'rho',rhos,snr),'rate','cee',method,i)
            cap=2 if scenario=='BIA' else 4
            ax.set_xlim(0,cap*1.04);ax.set_title(f'{scenario}, {snr} dB');ax.legend(fontsize=7.5)
    book.save('11_multi_snr_shared_source_tradeoff',fig,'固定 SNR、改变真实帧能量分配。对高 SNR 接近 QPSK 上限的速率使用固定横轴范围，不把极小的通信差距放大成巨大优势。')
    fig,axes=book.new(2,2,(11,8),'Actual-airtime tradeoffs, fixed DFT sequence')
    for row,scenario in enumerate(('BIA','TIM')):
        for ax,snr in zip(axes[row],(10,30)):
            for i,method in enumerate((scenario,'SIC','TIN')):
                book.line(ax,g.series(scenario,'dft',method,'rho',rhos,snr),'rate_native','cee',method,i)
            ax.set_xlim(0,(3 if scenario=='BIA' else 10)*1.04)
            ax.set_title(f'{scenario}, {snr} dB');ax.legend(fontsize=8)
    book.save('12_actual_airtime_tradeoff',fig,'按各格式实际发送时长计速率。直接叠加更短，在高 SNR 时可能拥有更高的原始比特/实际时隙上限；不隐藏编码扩展的时间代价。')
    book.finish()
    m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));m['figures']=book.entries
    root=Path(__file__).resolve().parents[1]
    m['plot_source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('isac/shared_source_plots.py','isac/plotting.py')}
    (out/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8')
    text=['# 共享原始符号的修正结果','',
          '本版不沿用旧版“每时隙新数据 + 单用户检测”的基线。两格式原始符号完全相同，通信均用相同有限 QPSK MAP 接收机。',
          '',
          '| 场景 | SNR | 编码共同窗口速率 | 直接叠加共同窗口速率 | 编码 BER | 直接叠加 BER |',
          '|---|---:|---:|---:|---:|---:|']
    for scenario in ('BIA','TIM'):
        for snr in snrs:
            a=g.point(scenario,'dft',main,snr,scenario);d=g.point(scenario,'dft',main,snr,'TIN')
            text.append(f'| {scenario} | {snr} dB | {a["rate"]:.5f} | {d["rate"]:.5f} | {a["ber"]:.5g} | {d["ber"]:.5g} |')
    text+=['','速率为 QPSK 互信息样本均值，不是已经部署某个纠错码后的吞吐量。共同窗口下上限分别为 2 和 4 bit/channel use。第 3、12 图另给出实际时长口径。',
           '','直接叠加占 4 个时隙，原 BIA/TIM 分别占 6/10 个；每个信息符号能量和每发射机感知能量相同。两种时间口径都展示，不将基线的较短时长隐去。',
           '','通信和感知分别使用不同 Monte Carlo 样本预算；各自实验内部对各方法共享原始符号、信道和噪声。图比较均值，不用于推导同帧联合中断概率。',
           '','不按性能好坏筛选点。通信检测器没有读入真实数据；真实比特仅用于 BER 与互信息评分。零观测误码不是证明真实 BER 为零。']
    (out/'结果说明.md').write_text('\n'.join(text),encoding='utf-8')
    print(f'Saved {len(book.entries)} shared-source figures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]/'results_shared_source')
    make_figures(p.parse_args().out)
