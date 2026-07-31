# Stream — EENN hardware-mapping experiments

This is the [Stream](https://github.com/KULeuven-MICAS/stream) DSE framework, extended and
pinned to the hardware-side experiments of:

> A. Zniber, A. Symons, O. Karrakchou, M. Verhelst, M. Ghogho,
> *"Hardware-Algorithm Co-Optimization of Early-Exit Neural Networks for Multi-Core Edge Accelerators"*,
> arXiv:2512.04705, 2026. <https://arxiv.org/abs/2512.04705>

Stream plays one role in that paper (Fig. 5, right-hand box): given an early-exit network
exported to ONNX and a multi-core accelerator description, it **optimizes the intra-core
mapping (LOMA) and the inter-core workload allocation (genetic algorithm)**, then reports
**energy and latency per early-exit stage**. Everything upstream — quantization-aware
training, accuracy, exit ratios, the NAS itself — lives in a separate repo; this repo
consumes its ONNX models and `*.stats` files and produces the hardware cost numbers.

---

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The three original paper scripts (`eenn_focus.sh`, `eenn_precision.sh`, `eenn_nas.sh`)
instead assume a conda env literally named `stream`; the hardware sweep runner takes the
interpreter from `$PYTHON` and defaults to `.venv/bin/python`.

Always run from the repository root: the accelerator parser resolves core yamls relative to
the literal path `stream/inputs/`.

---

## The hardware architecture

All three experiments target **one** accelerator: a quad-core Edge-TPU-like device,
described in `stream/inputs/eenn/hardware/edge_tpu_like_quad_core.yaml`.

```
stream/inputs/eenn/hardware/
├── edge_tpu_like_quad_core.yaml   # top level: core list + NoC
└── cores/
    ├── edge_tpu_like.yaml         # the 4 compute cores  (cores 0–3)
    ├── pooling.yaml               # pooling core         (core 4)
    ├── simd.yaml                  # SIMD / elementwise   (core 5)
    ├── offchip.yaml               # DRAM                 (core 6)
    └── tpu_like.yaml              # unused by the EENN experiments
```

**Top level** — 7 cores on a `2d_mesh` (`nb_rows: 2`, `nb_cols: 2`), NoC link bandwidth
32 bit/cycle, `unit_energy_cost: 0`; `pooling_core_id: 4`, `simd_core_id: 5`,
`offchip_core_id: 6`.

**Compute core** (`cores/edge_tpu_like.yaml`), replicated 4×:

| Component | Setting |
|---|---|
| Operational array | `dimensions: [D1, D2, D3, D4]`, `sizes: [8, 8, 4, 4]` → **1024 MAC/cycle** |
| MAC energy | 0.04 pJ |
| `rf_W` / `rf_O` | 32 bit register files (weights `I2`, partial sums `O`) |
| `sram_32KB` | 262 144 bit = 32 KiB weight buffer (`I2`), 512 bit r/w |
| `sram_2MB` | 16 777 216 bit = 2 MiB activation scratchpad (`I1`, `O`), 2048 bit r/w |
| `dram` | 64 bit/cycle r/w, 700/750 pJ |

No `dataflows` block is given for the compute core, so the spatial mapping is **searched**
per layer rather than fixed — array-dimension changes are picked up automatically.
The pooling and SIMD cores *do* fix their dataflow (`FX,3 × FY,3` and `K,64`).

> Note: the paper's appendix states 512 MAC/cycle per core; the committed YAML is
> `[8, 8, 4, 4] = 1024`. The 2 MiB activation scratchpad and 64 bit/cycle off-chip
> bandwidth do match the appendix.

**Mapping** (`stream/inputs/eenn/mapping/edge_tpu_like_quad_core.yaml`) pins ONNX op
types to cores: `Conv`/`Gemm`/default → cores `[0,1,2,3]` (the GA picks which one per
node); all `*Pool` → core `[4]`; `Add` → core `[5]`.

### Knobs for a compute/memory sweep

- **Compute**: `operational_array.sizes` (and `dimensions`) plus `multiplier_energy` in
  `cores/edge_tpu_like.yaml`.
- **Memory**: `sram_2MB.size` (activations), `sram_32KB.size` (weights), and the `r_bw` /
  `w_bw` / `r_cost` / `w_cost` of each level; `dram.r_bw`/`w_bw` for off-chip bandwidth.
- **Core count / NoC**: the `cores:` list and `graph.bandwidth` in the top-level YAML.
  The validator (`stream/parser/accelerator_validator.py`) enforces
  `nb_rows × nb_cols == len(cores) − (pooling + simd + offchip cores)`, and the special
  core ids must stay consistent — so changing the core count means editing the core list,
  `nb_rows`/`nb_cols`, the three `*_core_id` fields, **and** `core_allocation` in the
  mapping YAML.
- Point a run at a new architecture by editing the `accelerator` and `mapping_path`
  variables at the top of `main_stream_eenn.py` / `main_stream_eenn_nas.py`.

---

## Experiments

All three use `optimize_allocation_ga` in `mode="lbl"` (layer-by-layer, one layer per
stack) with a **64-generation × 64-individual** GA over the core allocation. Per-stage
energy/latency are accumulated in
`stream/opt/allocation/genetic_algorithm/fitness_evaluator.py`
(`save_eenn_data` / `write_eenn_data`), which splits the workload into backbone blocks and
exit classifiers by ONNX node name (`blocks.<i>/`, `classifiers.<i>/`) and records, per
exit stage: block latency, classifier latency, on-chip + off-chip CN energy, NoC
communication energy, the energy overhead of backbone work that overlaps a classifier, and
cumulative latency/energy/MAC fractions.

The workload is a 12-block MobileNetV2 backbone. A model is identified by the block
indices its intermediate exits are mounted after; the paper labels those positions with
letters (`0→A`, `1→B`, …), with the always-present final exit as `K`. So `-id 3,5,8` is
model **[D, F, I, K]**.

### 1 · Impact of quantization — paper §III-B, Table I, Fig. 2

Fixed model `[D, F, I, K]`, sweep of (backbone, classifier) precision. Precision is applied
in `stream/stages/parsing/onnx_model_parser.py::override_precision_workload`, which rescales
operand precisions and tensor sizes for every node — nodes whose name contains `classifier`
get `-pc`, all others get `-pb`.

```bash
mkdir -p outputs-eenn/precision_logs
./eenn_precision.sh          # 5 runs: (pb,pc) = (32,32) (8,8) (8,4) (4,8) (4,4)
python stream/visualization/eenn/plot_precision.py
```

`plot_precision.py` reads `outputs-eenn/precision_data/model_3_5_8_pb{pb}_pc{pc}.pickle`
and the hardcoded `EXIT_RATIOS` table, and emits `precision.pdf` — the two panels of Fig. 2
("Cumulative ET across stages" / "Average ET based on exit ratios").

### 2 · Impact of mounting points — paper §III-C, Fig. 3 and Fig. 4

Fixed 8-bit/8-bit precision, sweep over 13 four-exit models with identical backbone and
exit topology but different mounting points. The 13 `-id` triples in `eenn_focus.sh` are
exactly the 13 labelled points in Fig. 3, and each has an ONNX model under
`stream/inputs/eenn/workload/focus/model_<i>_<j>_<k>/`.

```bash
mkdir -p outputs-eenn/focus_logs
./eenn_focus.sh              # 13 runs, all at -pb 8 -pc 8
python stream/visualization/eenn/plot_focus.py            # -> Fig. 3
python stream/visualization/eenn/plot_mounting_points.py  # -> Fig. 4
```

- `plot_focus.py` (**Fig. 3**) scans the `focus/` model dirs, reads
  `outputs-eenn/focus_data/model_<i>_<j>_<k>_pb8_pc8.pickle` plus each model's
  `net_*_stats.stats` (for `top1_accuracy` and `exits_ratios`), and plots average accuracy
  vs. average ET for the whole family.
- `plot_mounting_points.py` (**Fig. 4**) zooms in on the 5 best models —
  `(0,2,6) (0,4,6) (0,4,8) (2,3,6) (3,5,8)` = `[A,C,G,K] [A,E,G,K] [A,E,I,K] [C,D,G,K]
  [D,F,I,K]`, the same set kept in `stream/inputs/eenn/workload/pareto/` — reading
  `outputs-eenn/mounting_points_data/pareto/model_<i>_<j>_<k>.pickle` and plotting ET per
  exit stage next to the exit-ratio-weighted average ET.

### 3 · Hardware-aware NAS evaluation — paper §V, Fig. 8

Stream evaluates the 83 EENN architectures produced by the NAS across its iterations. Each
lives in `stream/inputs/eenn/workload/nas/iter_<n>/net_<m>/` (iterations 0, 2–7; 10, 10, 27,
17, 7, 10, 2 models respectively) together with its `.stats` file. Unlike experiments 1–2,
the number of exits varies per model, so `main_stream_eenn_nas.py` takes a variable-length
`-id` list; the last entry is always `11` (the final exit, `K`).

```bash
./eenn_nas.sh                # 83 runs, 12 at a time, all at 8-bit/8-bit
python stream/visualization/eenn/plot_nas.py
```

`get_model_ids_for_nas.py` regenerates the `relpath → exit ids` table at the top of
`eenn_nas.sh` by inspecting each `model.onnx` for classifier nodes fed by a backbone block.

`plot_nas.py` walks the NAS model dirs, joins each model's per-stage pickle with its
`.stats` file, computes for every model the exit-ratio-weighted `avg_edp` and the
`edp_reduction` factor against the same backbone without early exits, and caches the joined
dataframe at `outputs-eenn/nas/df_cache.pickle` (**delete this file after re-running the
sweep**, or the plot will reuse stale numbers). Points are coloured by number of exits and
filtered to `avg_edp < EDP_UB`. Two configurations are provided at the top of the file:

- the block that is commented out (`accuracy` vs `avg_edp`, `ADD_PARETO = True`) reproduces
  **Fig. 8**, including the Pareto front and the star marking the model reported in Table II;
- the active block (`edp_reduction` vs `avg_edp`) is the ET-reduction variant.

Fig. 6 and Fig. 7 (per-NAS-iteration scatter and box plots) aggregate the same Stream
outputs but are produced on the NAS/training side, not in this repo.

---

## Where results land

Per run, `optimize_allocation_ga` writes `saved_cn_hw_cost.pickle` (the intra-core cost
LUT) and `scme.pickle` under `<output_path>/<experiment_id>/`, and the two main scripts
additionally dump `schedule.html` (Plotly timeline) and `memory.png`. Experiments 1–2 use
`outputs-eenn/focus/...`, experiment 3 uses `outputs-eenn/nas/<iter>/<net>/...`.

The per-exit-stage pickles that all the plot scripts consume are written separately, to
`f"{model_path}/model_<ids>.pickle"`. **`main_stream_eenn_nas.py` passes `model_path`** (the
model's own directory, which is where `plot_nas.py` looks), but **`main_stream_eenn.py` does
not** — so for experiments 1–2 that path is unset, and the published figures were made from
pickles collected by hand into `outputs-eenn/precision_data/`, `outputs-eenn/focus_data/`
and `outputs-eenn/mounting_points_data/pareto/`. Set `model_path` in `main_stream_eenn.py`
(or move the pickles) before running the focus/precision plot scripts.

Note that `skip_if_exists=True` in `main_stream_eenn_nas.py` reloads a cached `scme.pickle`
when one exists — which skips the GA and therefore does *not* regenerate the per-stage
pickle. Clear `outputs-eenn/nas/` when re-running with changed hardware.

---

## Heterogeneous multi-core case study (beyond the paper)

The paper evaluates a single accelerator. This case study runs the **six Pareto-optimal EENN
architectures of Fig. 8** on **nine quad-core accelerators** built from three core styles at an
identical compute and memory budget, to separate the effect of the hardware from the effect of
the network.

| core | array | MAC | on-chip | register-file wiring |
|---|---|---|---|---|
| `E` edge_tpu_like | 8×8×4×4 | 1024 | 2.032 MiB | weight RF shared over `[D3,D4]` (one read feeds 16 MACs) |
| `M` meta_like | 32×2×4×4 | 1024 | 2.095 MiB | same sharing, wider `D1`, split weight/activation banks |
| `Y` eyeriss_like | 35×30 | 1050 | 2.031 MiB | every RF private to one multiplier, extra output SRAM |

The nine systems are the three homogeneous designs plus six mixtures, all sharing the same
pooling, SIMD and off-chip cores, mesh and mapping. `sys_EEEE` is exactly the Fig. 8
accelerator and anchors the numbers. Definitions and the full rationale are in
`stream/inputs/eenn/hardware/casestudy/README.md`.

```bash
./eenn_casestudy.sh -j 12                                  # 9 systems x 6 workloads, 64x64 GA
cd stream/visualization/eenn && python plot_casestudy.py   # figures + tables
python plot_casestudy.py --exclude MMMM                    # zoomed, MMMM compresses the y-axis
```

The sweep writes ~170 MB of per-stage GA traces per family, of which the figures use one row per
exit stage. Only the aggregated CSVs are committed
(`casestudy_runs.csv`, `casestudy_stages.csv`, `casestudy_allocation.csv`, about 32 KB), and
`plot_casestudy.py` falls back to them when the raw outputs are absent, so every figure
regenerates from a clean checkout. Verified: the PDFs produced from the CSVs are byte-identical
to the raw-derived ones apart from the embedded timestamp.

### Results

`sys_EEEE` reproduces Fig. 8 (mean ET 432, range 374 to 492).

1. **The accelerator choice matters as much as the network choice.** Mean ET is 432 for the
   all-`E` system, 467 for all-`Y` and 660 for all-`M`, a spread of **1.53×**. It is a latency
   effect, not an energy one: the three stay within **4%** in energy. The cause is array
   utilization, median 21.9% / 19.7% / 10.1%, since one `M` array dimension is only two wide
   and cannot be filled by most backbone layers.
2. **Mixing core styles gains nothing.** The best mixture matches the anchor. Under `lbl`
   scheduling consecutive layers are data dependent and the cores never run concurrently
   (summed busy time is 0.96× the end-to-end latency), so the allocator sends up to **99.7%**
   of the computation to the fastest core type and leaves the rest idle. Exploiting
   heterogeneity requires several layers resident at once, i.e. layer-fused execution.
3. **Exit classifiers run below 1% array utilization on every core style.** Their pooling and
   fully connected operators are too small to fill a 1000-MAC array, so the early-exit overhead
   is structural and cannot be absorbed by the compute core.
4. **The EENN ranking transfers.** The ordering of the six architectures is preserved up to
   seven inversions, all between architectures within 2.34% of each other, against a measured
   search variability of 1.2%. The two-exit model is cheapest on 8 of 9 systems and the
   seven-exit model most expensive on all 9.

Differences below roughly 1.5% are at the level of the allocation search and should not be
interpreted: re-running one configuration four times reproduced a value that differed from the
sweep by 1.2%.

---

### Fixes this required

The repo as committed could not run any EENN experiment, and heterogeneous accelerators were not
modelled at all. Six defects, all fixed:

1. **`cores/{pooling,simd,offchip}.yaml` used an obsolete `multipliers:` key** where
   `zigzag-dse==3.6.3` requires `operational_array:`, introduced by the same commit that added
   the figure scripts. Every accelerator referencing them, including the paper's own
   `edge_tpu_like_quad_core.yaml`, failed to parse.
2. **Core files are referenced by bare filename**, which makes the validator walk
   `stream/inputs/` and take the first match. Two different `pooling.yaml` exist, 9 units and 72
   units, so which one is used depends on filesystem order. Fig. 8 was produced with the 72-unit
   core; picking the 9-unit one puts every number **2.7× high**. Isolated by measurement: 1013
   vs 375 on the same workload, with SIMD and off-chip making no difference (1013 vs 1015). The
   case-study accelerators pin every core by explicit path.
3. **`main_stream_eenn.py` never passed `model_path`**, so the per-stage pickle path became the
   literal string `"None/model_<ids>.pickle"` and `pickle_save` raised `FileNotFoundError` after
   the full GA had run, losing the entire run.
4. **`-id` dropped the final exit**, so the fitness evaluator recorded one stage too few and the
   last backbone blocks and final classifier were silently excluded. The entry point now appends
   `--last_block` and auto-detects the NAS convention where the final exit is already present.
5. **A single-compute-core accelerator recorded nothing**: with no allocation freedom the GA is
   skipped and the one evaluation called `write_eenn_data()` without ever calling
   `save_eenn_data()`. `get_fitness` now always records the allocation it evaluated.
6. **Heterogeneous cores were collapsed into one.** zigzag's `SpatialMappingGeneratorStage`
   resolves the core with `layer.core_allocation[0]`, which Stream never set, so every
   `(node, core)` pair was evaluated on core 0's hardware. Invisible on a homogeneous
   accelerator; on a heterogeneous one every core reported the core-0 cost. After also calling
   `set_core_allocation(core_id)`, all 39 layers of a test workload differentiate and the
   per-core costs match the homogeneous look-up tables exactly.

## Other files

`main_stream_ga.py`, `main_stream_co.py` and `main_testing_*.py` are the upstream Stream
examples and regression entry points (ResNet-18 on TPU-like / Eyeriss-like architectures
from `stream/inputs/examples/` and `stream/inputs/testing/`). They are unrelated to the
EENN paper. General Stream documentation: <https://kuleuven-micas.github.io/stream/>.
