import shutil
import subprocess
from pathlib import Path
import fileinput
import sys
import os
import numpy as np

ROOT = Path('.').resolve()
SRC = ROOT / 'SourceCode' / 'Power' / 'ftt_p_main.py'
BACKUP = ROOT / 'SourceCode' / 'Power' / 'ftt_p_main.py.bak'
IO = ROOT / 'IO_Output'
SENS_DIR = IO / 'sensitivity_runs'
SENS_DIR.mkdir(parents=True, exist_ok=True)

runs = [
    ('base', None, 0.0, 1.0),
    ('discount_low', 0.07, 0.0, 1.0),
    ('discount_high', 0.13, 0.0, 1.0),
    ('selfcons_minus', None, -0.05, 1.0),
    ('selfcons_plus', None, 0.05, 1.0),
    ('export_0.8', None, 0.0, 0.8),
    ('export_1.2', None, 0.0, 1.2),
]

# helper to patch discount_rate in source file
def patch_discount(tmp_value):
    if tmp_value is None:
        return
    # backup original
    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
    text = SRC.read_text(encoding='utf-8')
    # replace the block setting discount_rate and elec_price_growth
    new_text = text.replace('\n            # Discounted benefit\n            discount_rate = 0.1\n            elec_price_growth = 0.02\n\n            lifetime_pv =',
                            f"\n            # Discounted benefit\n            discount_rate = {tmp_value}\n            elec_price_growth = 0.02\n\n            lifetime_pv =")
    SRC.write_text(new_text, encoding='utf-8')

# restore backup
def restore_src():
    if BACKUP.exists():
        shutil.move(str(BACKUP), str(SRC))

# run a single run by importing ModelRun and modifying inputs

def run_and_collect(runname, delta_selfcons, mult_export):
    outdir = SENS_DIR / runname
    outdir.mkdir(parents=True, exist_ok=True)
    # use a subprocess to run a small Python snippet that creates model, patches inputs, runs
    py = f"""
import sys
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
import numpy as np
from SourceCode.model_class import ModelRun
from pathlib import Path
m = ModelRun()
# scenarios keys
scen = list(m.input.keys())[0]
# Patch RSSC and RSFT across all entries in the input for scenario
if 'RSSC' in m.input[scen]:
    arr = m.input[scen]['RSSC']
    # RSSC shape: (regions, techs, 1)
    arr = arr.copy()
    arr[:,:, :] = np.clip(arr + {delta_selfcons}, 0.0, 1.0)
    m.input[scen]['RSSC'] = arr
if 'RSFT' in m.input[scen]:
    arr2 = m.input[scen]['RSFT']
    arr2 = arr2.copy()
    arr2 = arr2 * {mult_export}
    m.input[scen]['RSFT'] = arr2
m.run()
print('DONE')
""".strip()
    # write to a temp file
    runner = ROOT / 'tools' / f'_sens_run_{runname}.py'
    runner.write_text(py)
    cmd = [sys.executable, str(runner)]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    runner.unlink()
    # move outputs
    for fname in ['rooftop_trace_au.txt','rooftop_trace_au_series.csv','rooftop_trace_au_plot.png']:
        src = IO / fname
        if src.exists():
            dst = outdir / fname
            shutil.move(str(src), str(dst))

try:
    for name, disc, delta_self, mult_export in runs:
        print('Running', name)
        if disc is not None:
            patch_discount(disc)
        else:
            # restore original if exists
            if BACKUP.exists():
                restore_src()
        run_and_collect(name, delta_self, mult_export)
    print('All runs complete')
finally:
    restore_src()

print('Sensitivity runs finished; outputs in', SENS_DIR)
