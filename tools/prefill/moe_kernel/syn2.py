import sys, os
import moe_fp4
if os.environ.get("NSL"):
    moe_fp4.NS = [int(v) for v in os.environ["NSL"].split(",")]
os.environ["MOE_FP4_VERBOSE"] = "1"
moe_fp4._ext()
sys.argv = sys.argv
exec(open("syn.py").read().replace("import moe_fp4\n", ""))
