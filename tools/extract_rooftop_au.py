import re
import csv
from pathlib import Path
import matplotlib.pyplot as plt

inp = Path('IO_Output/rooftop_trace_au.txt')
out_csv = Path('IO_Output/rooftop_trace_au_series.csv')
out_png = Path('IO_Output/rooftop_trace_au_plot.png')

if not inp.exists():
    raise SystemExit('Input file not found: ' + str(inp))

years = []
prich = []
cost_roof = []
cost_grid = []
std_roof = []
share = []
mewg = []
mewl = []
mewk = []
mews = []
rooftop_cap_share = []

with inp.open('r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

i = 0
while i < len(lines):
    m = re.match(r'^Year\s+(\d+) \| Region .*', lines[i])
    if m:
        year = int(m.group(1))
        years.append(year)
        # next lines contain fields
        # PRICH etc
        i += 1
        # read up to next Year or EOF
        block = []
        while i < len(lines) and not lines[i].startswith('Year '):
            block.append(lines[i])
            i += 1
        # join block and parse
        txt = ' '.join(block)
        # PRICH
        m2 = re.search(r'PRICH=([\d\.,\-]+)', txt)
        prich.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'cost_roof=([\d\.,\-]+)', txt)
        cost_roof.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'cost_grid=([\d\.,\-]+)', txt)
        cost_grid.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'std_roof=([\d\.,\-]+)', txt)
        std_roof.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'share=([\d\.,\-]+)', txt)
        share.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'MEWG=([\d\.,\-]+)', txt)
        mewg.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'MEWL=([\d\.,\-]+)', txt)
        mewl.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'MEWK=([\d\.,\-]+)', txt)
        mewk.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'MEWS=([\d\.,\-]+)', txt)
        mews.append(float(m2.group(1).replace(',', '')) if m2 else None)
        m2 = re.search(r'rooftop_cap_share=([\d\.,\-]+)', txt)
        rooftop_cap_share.append(float(m2.group(1).replace(',', '')) if m2 else None)
    else:
        i += 1

# write CSV
with out_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Year','PRICH','cost_roof','cost_grid','std_roof','share','MEWG','MEWL','MEWK','MEWS','rooftop_cap_share'])
    for row in zip(years, prich, cost_roof, cost_grid, std_roof, share, mewg, mewl, mewk, mews, rooftop_cap_share):
        writer.writerow(row)

# plot rooftop_cap_share and share
plt.figure(figsize=(10,4))
plt.plot(years, [s*100 for s in rooftop_cap_share], label='rooftop_cap_share (%)')
plt.plot(years, [s*100 for s in share], label='household_share (%)')
plt.xlabel('Year')
plt.ylabel('Share (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png)
print('Wrote', out_csv, out_png)
