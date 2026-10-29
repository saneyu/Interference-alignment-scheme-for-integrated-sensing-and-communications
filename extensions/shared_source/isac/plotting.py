from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COLORS = ['#52788D', '#BC8450', '#668C7A', '#A35F69', '#777777', '#8B83A3']


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler(color=COLORS), 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'savefig.dpi': 220, 'axes.grid': True,
        'grid.alpha': .2, 'axes.axisbelow': True})


def stats(samples):
    a = np.asarray(samples, float)
    mean = float(a.mean())
    half = 1.96*float(a.std(ddof=1))/np.sqrt(len(a))
    return mean, max(0., mean-half), mean+half


def line(ax, x, samples, label, log=False, color=None, marker=None):
    summary = np.array([stats(a) for a in samples])
    index = len(ax.lines)
    opts = {'label': label, 'marker': marker or ['o','s','^','D','v','x'][index%6],
            'linestyle': ['-','--','-.',':','-','--'][index%6],
            'markersize': 3.5, 'linewidth': 1.5}
    if color:
        opts['color'] = color
    drawn, = ax.plot(x, summary[:, 0], **opts)
    if log:
        ax.set_yscale('log')
    ax.fill_between(x, np.maximum(summary[:, 1], 1e-14), summary[:, 2],
                    color=drawn.get_color(), alpha=.13, linewidth=0)
    return summary


class Gallery:
    def __init__(self, out):
        style()
        self.out = Path(out)
        for name in ('figures', 'data', 'raw'):
            (self.out/name).mkdir(parents=True, exist_ok=True)
        self.pdf = PdfPages(self.out/'all_figures.pdf')
        self.entries = []

    def save(self, stem, fig, rows, caption):
        for fmt in ('png', 'pdf', 'svg'):
            fig.savefig(self.out/'figures'/f'{stem}.{fmt}', bbox_inches='tight')
        self.pdf.savefig(fig, bbox_inches='tight')
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (self.out/'data'/f'{stem}.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
        self.entries.append({'id': stem, 'caption': caption})
        plt.close(fig)

    def finish(self):
        self.pdf.close()
        fig, axes = plt.subplots(6, 3, figsize=(18, 22), layout='constrained')
        for ax, entry in zip(axes.flat, self.entries):
            ax.imshow(plt.imread(self.out/'figures'/f"{entry['id']}.png"))
            ax.axis('off'); ax.set_title(entry['id'], fontsize=9)
        for ax in axes.flat[len(self.entries):]:
            ax.axis('off')
        fig.savefig(self.out/'overview.png', dpi=160)
        plt.close(fig)
        html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>ISAC results</title>',
            '<style>body{font:16px system-ui;max-width:1150px;margin:36px auto;background:#fafafa;color:#243746}article{background:white;padding:24px;margin:24px 0;border:1px solid #ddd;border-radius:8px}img{width:100%}a{color:#52788d}</style>',
            '<h1>BIA / TIM 仿真结果</h1><p>独立复基带重实现；统计与限制请结合 README 和 audit 阅读。</p><a href="all_figures.pdf">下载全部图 PDF</a>']
        for e in self.entries:
            name = e['id']
            html.append(f'<article><h2>{name}</h2><img src="figures/{name}.png"><p>{e["caption"]}</p><a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.pdf">PDF</a> · <a href="data/{name}.csv">CSV</a></article>')
        html.append('</html>')
        (self.out/'index.html').write_text('\n'.join(html), encoding='utf-8')


def rows_for(x, samples, method, xname, **extra):
    rows = []
    for xi, a in zip(x, samples):
        mean, low, high = stats(a)
        rows.append({xname: float(xi), 'method': method, 'mean': mean,
                     'ci95_low': low, 'ci95_high': high, 'n': len(a), **extra})
    return rows
