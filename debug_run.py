# debug_run.py
import traceback
from ftt_source.model_class import RunFTT

try:
    m = RunFTT(settings={"enable_modules": "FTT-H", "scenarios": "S0", "simulation_end": "2030"})
    print("Inputs loaded OK")
    m.run()
    print("Run OK")
except Exception:
    traceback.print_exc()


# find_bad_dims.py
from ftt_source.support import titles_functions as t, dimensions_functions as d

titles = t.load_titles()
dims = d.load_dims()[0]
titles["TIME"] = ()                      # TIME is added at run time

for var, dd in dims.items():
    bad = [x for x in dd if x not in titles and x != "NA"]
    if bad:
        print(f"{var!r}: dims={dd}  -> not in titles: {bad}")