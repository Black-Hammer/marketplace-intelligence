# ── CELL 18: Run ch6_analysis.py ─────────────────────────────────────────
!pip install -q tabulate
!python ch6_analysis.py \
    --indir  data/processed \
    --outdir ch6_tables \
    --maxlag 4
print('\n✓ ch6_analysis complete — tables in ch6_tables/')
import os; print(os.listdir('ch6_tables'))

# ── CELL 19: Visualise table_6_4_forecasting ──────────────────────────────
import matplotlib.pyplot as plt
import re

tbl_path = 'ch6_tables/table_6_4_forecasting.md'
rows = []
with open(tbl_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('|') and '---' not in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            rows.append(cells)

header, data = rows[0], rows[1:]
df_viz = pd.DataFrame(data, columns=header)
for col in ['MAE','RMSE','MAE %impr vs control']:
    df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')

COLS = {'MAE':'MAE (₦)','RMSE':'RMSE (₦)','MAE %impr vs control':'MAE % Improvement'}
PAL  = ['#4338CA','#0D9488','#94A3B8','#64748B','#EA580C']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chapter 6 — Forecasting Model Comparison', fontsize=15, fontweight='bold')
for ax, (col, label) in zip(axes, COLS.items()):
    vals = df_viz[col]
    bars = ax.bar(df_viz['Model'], vals, color=PAL[:len(df_viz)],
                   edgecolor='white', linewidth=0.8)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    ax.spines[['top','right']].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+abs(vals.max()*0.01),
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    if col == 'MAE %impr vs control':
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig('ch6_tables/figure_6_forecasting_comparison.png',
            dpi=200, bbox_inches='tight')
plt.show()
print('✓ figure saved to ch6_tables/')

# ── CELL 20: Download all outputs ─────────────────────────────────────────
from google.colab import files
import zipfile, os

with zipfile.ZipFile('chapter6_results.zip','w') as z:
    for folder in ['data/processed','ch6_tables']:
        for f in os.listdir(folder):
            z.write(os.path.join(folder,f), os.path.join(folder,f))

files.download('chapter6_results.zip')
print('✓ chapter6_results.zip downloaded')
