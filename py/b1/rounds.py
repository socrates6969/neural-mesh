import sys, json, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from battery import *
from gate import compare
D = Data()
import atexit; atexit.register(D.save_extra)
ALLS = TUNE + HELD + CONF
def base_tab():
    b = json.load(open(f"{RES}/b1_baseline.json"))["hybrid"]
    return {(k[0], int(k[1:])): v for k, v in b.items()}
def cand(seeds=ALLS, **kw):
    return run_battery(D, seeds, with_exec=False, **kw)
def tune_mean(tab, ax):
    return float(np.mean([tab[(w, s)][ax] for w in WORLDS for s in TUNE]))
KN = ["domain_r1", "domain_mrr", "domain_route", "general_info", "comprehension"]
