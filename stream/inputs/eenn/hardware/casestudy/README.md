# Heterogeneous multi-core case study

Nine 4-compute-core accelerators built from three core styles, for running the Figure 8
Pareto-optimal EENN architectures. Every system has the **same compute and memory budget** and
uses the **same pooling, SIMD and off-chip cores, the same 2x2 mesh and the same mapping** as
the accelerator in the paper. The only thing that varies is which compute core sits in each of
the four slots.

The point of the study is that the paper's accelerator is multi-core but *homogeneous*: four
identical cores. Broadening the space to mixed core types, at an unchanged budget, asks whether
a heterogeneous fabric deploys the NAS-found EENN architectures better than any homogeneous one.

## Anchoring

`sys_EEEE.yaml` is the accelerator used for Figure 8. It references the original
`cores/edge_tpu_like.yaml`, `cores/pooling.yaml`, `cores/simd.yaml` and `cores/offchip.yaml`
unchanged, so it is not a copy that can drift. Verified against
`hardware/edge_tpu_like_quad_core.yaml`: identical core set. Its results should reproduce the
published numbers, and every other system is read relative to it.

All nine use `mapping/edge_tpu_like_quad_core.yaml` (Conv/Gemm to cores 0-3 with the allocator
choosing, all pooling to core 4, Add to core 5) and `mode="lbl"`, matching Figure 8. In `lbl`
mode a single inference is dependency-serialised, so the four cores provide **allocation
freedom rather than concurrency**: measured on the anchor, the sum of all core busy time divided
by total latency is 0.96. Heterogeneity therefore pays off through specialisation, by letting
each layer land on the core that suits it, not through parallelism.

## The three core styles

What separates them is how the lowest memory level is wired to the compute array, which ZigZag
expresses as `served_dimensions`: the array dimensions that a *single instance* of a memory
level spans. A level serving `[D3, D4]` is instantiated once per `(D1, D2)` pair, so one read
feeds all 16 multipliers beneath it. A level serving `[]` is instantiated once per multiplier,
so every MAC pays its own read but gets private storage in exchange.

| core | file | array | MAC | on-chip | register-file wiring |
|---|---|---|---|---|---|
| `E` edge_tpu_like | `../cores/edge_tpu_like.yaml` (unchanged) | 8x8x4x4 | 1024 | 2.032 MiB | weight RF `[D3,D4]`, output RF `[D2]` |
| `M` meta_like | `cores/meta_like.yaml` | 32x2x4x4 | 1024 | 2.095 MiB | weight RF `[D3,D4]`, output RF `[D2]`, split W/A banks |
| `Y` eyeriss_like | `cores/eyeriss_like.yaml` | 35x30 | 1050 | 2.031 MiB | all three RFs `[]` (private per PE), extra output SRAM |

Budgets are within +3.1% (meta) and -0.05% (eyeriss) of the anchor on memory, and +0% / +2.5%
on MAC count. All three use 0.04 pJ per MAC and leave the spatial mapping to the mapper, so no
system is advantaged by a hand-picked unrolling.

### What was changed in the non-anchor cores, and why

**`meta_like`** is the published `examples/hardware/cores/meta_prototype.yaml` with two changes:
the obsolete `multipliers:` key renamed to `operational_array:` (the published file does not
parse at all under `zigzag-dse==3.6.3`), and an in-core DRAM identical to the anchor's added so
it is not penalised by having to reach off-chip over the NoC. Array, memory sizes and energies
are untouched, and its MAC energy was already 0.04 pJ.

**`eyeriss_like`** is Eyeriss-style *wiring at this budget*, not Eyeriss as published, and the
paper should describe it that way. The published core is 14x12 = 168 MAC, 1.09 MiB and 0.5 pJ
per MAC, a different size and a different technology node; comparing it directly would measure
those rather than the architecture. What is preserved is the defining property: every register
file private to one multiplier, plus the dedicated output SRAM level. Changes:

- array `35x30 = 1050 MAC`, the closest multiple of the published 7:6 aspect ratio to 1024
- MAC energy 0.04 pJ, matching the other two
- register-file sizes unchanged (64 B / 64 B / 16 B per PE); energies from the relation
  `cost = 3.711e-4 * bandwidth * sqrt(size_bits)`, calibrated on meta's own register files
  (reproduces its `rf_1B` as 0.008 against an actual 0.01, and `rf_2B` as 0.024 against 0.02)
- SRAM capacities set so total on-chip is 2.031 MiB, keeping the published three-level split;
  energies scaled from the nearest reference level by `sqrt(size)`, namely `sram_main` from the
  anchor's `sram_2MB`, `sram_128KB_W` from meta's `sram_64KB`, `sram_16KB_O` from the anchor's
  `sram_32KB`
- in-core DRAM identical to the anchor's
- no `dataflows` block; the published one declares `K,16 / C,16`, which does not even fit the
  published 14x12 array

## The nine systems

Full space is 3 styles in 4 slots = 15 distinct compositions. These nine were chosen to answer
four questions without running all of them.

| file | cores | what it tests |
|---|---|---|
| `sys_EEEE` | E E E E | **anchor, = Figure 8 hardware** |
| `sys_MMMM` | M M M M | homogeneous reference |
| `sys_YYYY` | Y Y Y Y | homogeneous reference |
| `sys_EEYY` | E E Y Y | the two most structurally different styles, evenly split |
| `sys_EEMM` | E E M M | **control**: two *similar* styles. Little gain expected |
| `sys_MMYY` | M M Y Y | the third pair, no anchor core present |
| `sys_EEEY` | E E E Y | is a single specialist core enough? |
| `sys_EYYY` | E Y Y Y | does the ratio matter, or only presence? |
| `sys_EEMY` | E E M Y | all three styles available to the allocator |

Rows 1-3 give each style's standalone level. Rows 4-6 are the complete set of pairwise even
splits, the cleanest test of complementarity; `sys_EEMM` is deliberately a control, since `E`
and `M` are both 4D arrays sharing their weight RF across `[D3,D4]`, so if it gains nothing
while `sys_EEYY` does, the benefit comes from genuine architectural difference rather than from
mixing per se. Rows 7-8 form a ratio ladder with row 4 (3+1, 2+2, 1+3). Row 9 is the only
composition with all three styles present.

The six 3+1 compositions not listed were dropped as the least informative per run; once the
pairwise rows show which styles are complementary, the missing ladders are largely predictable.

**Placement within the mesh is fixed, not swept.** Cores fill slots in order, and the 2x2 mesh
reshapes column-major to `[[c0, c2], [c1, c3]]`, so links are 0-1, 0-2, 1-3, 2-3 and the
diagonals 0-3 and 1-2 are not directly connected. Placement is therefore a real variable in
principle, but it measured as a **0.25%** effect in an earlier heterogeneous sweep, because
`lbl` mode generates almost no concurrent inter-core traffic. It does not pay for the extra runs.

## Running

Core paths inside these files are explicit rather than bare filenames, because a bare name makes
the validator walk `stream/inputs/` and take the first match, and `pooling.yaml`, `simd.yaml`,
`offchip.yaml` and `tpu_like.yaml` each exist in more than one hardware folder. Always run from
the repository root.

```bash
python main_stream_eenn.py -id <exits> -w <path/to/model.onnx> \
  -hw stream/inputs/eenn/hardware/casestudy/sys_EEEE.yaml \
  -map stream/inputs/eenn/mapping/edge_tpu_like_quad_core.yaml \
  -o outputs-eenn/hw_casestudy -g 64 -i 64
```

The workloads are the six Figure 8 Pareto-optimal architectures:

| Figure 8 label | directory | exits | accuracy |
|---|---|---|---|
| `6_11_iter_0` | `workload/nas/iter_0/net_3` | 2 | 87.28% |
| `2_8_11_iter_0` | `workload/nas/iter_0/net_8` | 3 | 87.35% |
| `0_4_5_7_11_iter_0` | `workload/nas/iter_0/net_1` | 5 | 88.04% |
| `2_3_4_5_9_11_iter_3` | `workload/nas/iter_3/net_3` | 6 | 88.40% |
| `0_1_5_8_9_11_iter_2` | `workload/nas/iter_2/net_7` | 6 | 88.51% |
| `2_3_4_5_6_9_11_iter_6` | `workload/nas/iter_6/net_0` | 7 | 88.54% |

Their `-id` is the full exit list including the final exit at block 11, which is the convention
`main_stream_eenn_nas.py` uses and which `main_stream_eenn.py` now detects automatically.
