# Measurement data — "When Does Layer–Timestep Separability Hold?"

Every number in the paper's figures and tables comes from these files. One row per
measured observation, no aggregation applied.

## Read this first — three caveats that affect what you can plot

1. **`fid_results.csv` contains a known-wrong uniform arm at budget 6.0, and the
   fix lives in a second file.**
   The uniform baseline picks a single bit-width from {1,2,4,8}. The original code
   tie-broke budget 6 toward W4, so the row `arm=uniform, budget=6.0` is a **4-bit**
   model competing against allocators that spend ~5.97 bits. It is not a matched
   baseline and must not be plotted as one. No uniform arm sits *at* 6 bits, so the
   honest reporting is the bracket: W4 at 385.1 from that row, and W8 at
   **30.70 ± 0.93** from `fid_uniform_w8.jsonl` (3 seeds, 2000 samples, same
   pipeline). Table IV and Fig. 5b use the bracket. The `separable` and `measured`
   rows at B=6 were always unaffected and valid.

2. **The two grids are not matched in statistical power.**
   `grid_cifar.csv`: 256 paired images per cell, bit-widths {1,2,4,8}, 64 cells.
   `grid_celebahq.csv`: 64 paired images per cell, bit-widths {1,4}, 32 cells.
   256×256 sampling is far more expensive. Effect sizes on CelebA-HQ are large
   (noise/detail ratios 2.2–22.3) so the regime contrast is unlikely to be sampling
   noise, but do not present the two as equally powered.

3. **The coarse layer partition pools 39 of 63 convolutions into `up_all`.**
   A pilot measured two blocks inside that group differing by **102×** at W1 in the
   noise band (`u0` 0.00427 vs `u3` 0.43496). The CIFAR interaction share is therefore
   likely a **lower bound**. A nine-group control (one group per U-Net block) is
   implemented but has not completed.

## Files

| file | rows | what it is |
|---|---|---|
| `grid_cifar.csv` | 64 | CIFAR-10 DDPM joint grid: one row per (layer group × timestep band × bit-width) |
| `grid_celebahq.csv` | 32 | Same protocol on CelebA-HQ 256 |
| `anova_split.csv` | 6 | Two-way main-effects decomposition per model per bit-width |
| `temporal_profiles.csv` | 24 | Per-group damage at both trajectory ends + direction of the gradient |
| `composition_trials.csv` | 32 | Multi-cell configs: additive prediction vs measurement |
| `allocation_paired_rms.csv` | 24 | Allocation arms scored by the paired proxy, 3 noise streams |
| `fid_results.csv` | 30 | FID by arm and budget, 3 seeds (see caveat 1) |
| `cross_hardware.csv` | 47 | Same cells on T4 vs RTX 2050 |
| `session_replication.csv` | 64 | Same notebook, two independent Kaggle sessions |

## Conventions

**`damage_rms`** — the paired metric. Both the quantized configuration and the
full-precision reference are sampled from an *identical* fixed noise stream; damage is
the mean per-pixel RMS difference between matched images, in [0,1] pixel space. Lower
is closer to full precision. **It is not a quality metric** — see §8 of the paper.

**`timestep_band`** — `b0_detail` = t/T ∈ [0, 0.25) is the *end* of sampling (final
detail); `b3_noise` = [0.75, 1.0] is the *start* (near-pure noise). Sampling runs
t: 1 → 0, so plots should read detail → noise left to right, or be labelled explicitly.

**`layer_group`** — `down_early` = down_blocks.0–1 (11 convs), `down_late` =
down_blocks.2–3 (9), `mid` = mid_block (4), `up_all` = up_blocks.0–3 (39).

**ANOVA shares** — fractions of total log-variance from the main-effects model
`log D[g,t] = mu + a_g + c_t` fitted by row/column means. `share_interaction` is the
residual. This is deliberately *not* a rank-1 outer product of the log matrix: that
model has a free per-group multiplier and can itself reproduce a temporal sign flip,
which would hide the effect under test.

**Noise floor** — `noise_floor_rms` is 0.0 throughout. Under fixed execution conditions
(cuDNN algorithm selection pinned, TF32 off, identical batch size) an identical re-draw
is bit-identical. This is reproducibility under those conditions, not exactness in
general: comparing re-draws at a *different batch size* produced a spurious 9.3×10⁻⁴
drift, the same order as a weakly sensitive cell.

## Headline numbers, for cross-checking

```
CIFAR-10 DDPM     layer 87% / timestep  2% / interaction 11.2%   (W1)
                  2 of 4 groups rise toward noise  -> sign reversal
CelebA-HQ 256     layer 58% / timestep 38% / interaction  4.0%   (W1)
                  4 of 4 groups rise toward noise  -> no reversal

composition       r = 0.888, slope 0.53, mean measured/predicted 0.72  (n = 32)
allocation (proxy) +23.4% mean; r = 0.968 vs number of differing cells
FID at B = 6      separable 43.35 ± 0.20   measured 98.20 ± 0.51   FP32 28.44 ± 0.25
cross-hardware    r = 0.9991 on log damage, 47 cells
session repeat    bit-identical, max abs diff 0.000, 64 cells
```

## Setup

CIFAR: `google/ddpm-cifar10-32`, 63 quantizable convs, DDIM 50 steps over a 1000-step
schedule. CelebA-HQ: `google/ddpm-celebahq-256`, same sampler, beta schedule taken from
the model's own config. **Weights only** — activations are not quantized, and `conv_in`
/ `conv_out` stay FP32. Per-output-channel symmetric quantization with an MSE-optimal
clipping range; 1-bit uses sign with a per-channel mean-magnitude scale. All grid and
allocation results on an NVIDIA T4.
