import pickle
from zigzag.classes.cost_model.cost_model import CostModelEvaluation

pickle_filepath_lbl = "outputs\saved_cn_hw_cost-heterogeneous_quadcore-resnet18-hintloop_backup.pickle"
pickle_filepath_lf = "outputs\saved_cn_hw_cost-heterogeneous_quadcore-resnet18-hintloop_oy_all.pickle"
SCALE_FACTOR_LF = 112

with open(pickle_filepath_lbl, "rb") as handle:
        node_hw_performances_lbl = pickle.load(handle)
with open(pickle_filepath_lf, "rb") as handle:
        node_hw_performances_lf = pickle.load(handle)

print("Layer-by-layer")
node = next((n for n in node_hw_performances_lbl if n.id == (0, 0)))
core = next((c for c in node_hw_performances_lbl[node] if c.id == 2))
cme = node_hw_performances_lbl[node][core]
b = 0
for op, bd in cme.energy_breakdown.items():
    s = [f"{i:.3e}" for i in bd]
    a = sum([i for i in bd])
    b += a
    print(op, s, f"{a:.3e}")
print(f"{b:.3e}")
f"{cme.energy_total:.3e}"

print("\nLayer-fused")
node = next((n for n in node_hw_performances_lf if n.id == (0, 0)))
core = next((c for c in node_hw_performances_lf[node] if c.id == 2))
cme = node_hw_performances_lf[node][core]
b = 0
for op, bd in cme.energy_breakdown.items():
    s = [f"{SCALE_FACTOR_LF*i:.3e}" for i in bd]
    a = sum([SCALE_FACTOR_LF*i for i in bd])
    b += a
    print(op, s, f"{a:.3e}")
print(f"{b:.3e}")
f"{SCALE_FACTOR_LF*cme.energy_total:.3e}"
pass
