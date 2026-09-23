# When Does Layer–Timestep Separability Hold? Measuring Sensitivity Structure in Mixed-Precision Diffusion Quantization

%% AUTHOR
Munal Singh¹, Marina Thottakkara¹, Kashikaa¹, Deepa D²
¹Student, School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India
²Professor, School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

%% ABSTRACT
Mixed-precision quantizers for diffusion models allocate bit-width along two axes, which
layer and where on the denoising trajectory, and summarise sensitivity along one axis with
the other marginalised away. Whether that factorisation is adequate has not been measured.
We give a controlled protocol that measures the joint structure directly: a paired probe
quantizes exactly one (layer group, timestep band) cell, holds everything else at full
precision, and samples from a fixed noise stream, so an identical re-draw of the reference
differs by exactly zero under fixed execution conditions. Applied to two diffusion models,
the protocol finds two different regimes. On a pixel-space CIFAR-10 DDPM the timestep main
effect accounts for 1.7–2.7% of log-damage variance while the layer-by-timestep interaction
accounts for 8.6–16.0%, because the temporal sensitivity gradient reverses sign between
network regions and cancels when marginalised. On a CelebA-HQ 256 denoiser the ordering
inverts: the timestep main effect reaches 37.9–51.0% and the interaction falls to 4.0–10.1%.
Separability is therefore a property of a particular model rather than of diffusion models
in general. We further report a negative result. An allocation derived by minimising our
paired trajectory metric reduces that metric by 23.4% on average at matched average
bit-width, yet at the only budget producing recognisable images it is worse on FID than the
separable allocation it was meant to beat, 98.2 against 43.3. A reproducible
trajectory-deviation proxy is not a substitute for the generation objective, and sensitivity
proxies should be validated against it before they are optimised.

%% KEYWORDS
diffusion models, post-training quantization, mixed precision, sensitivity analysis,
evaluation metrics, reproducibility, negative results

%% S1. Introduction

A diffusion model generates by running a denoiser many times, so inference cost is the cost
of one pass times the number of steps. Quantization attacks the first factor. Because the
denoiser is invoked under different conditions along the trajectory, near-pure noise at one
end and near-final detail at the other, it is natural to allocate precision not only per
layer but per timestep.

Existing methods do this by factorisation. Qua2SeDiMo [1] and MPQ-DM [5] summarise
sensitivity per layer. TMPDiff [2] assigns one bit-width per timestep, shared across all
layers. DiffPro [3] and TMPQ-DM [4] combine a per-layer score with a separate timestep
criterion. In each case the joint quantity D(g,t) is replaced by terms that depend on one
axis at a time.

Whether that replacement is safe is an empirical question. The axes are not independent by
construction: a layer's contribution at step t depends on the activations reaching it, and
those activations are what changes along the trajectory. This paper measures the joint
structure rather than assuming a form for it. Our contributions are as follows.

1. A controlled protocol for measuring D(g,t) by isolated intervention, with a paired
   metric whose within-device reproducibility is exact under fixed execution conditions.
2. Two regimes, measured. On a CIFAR-10 DDPM the interaction exceeds the timestep main
   effect by 4.4 to 6.4 times and the temporal gradient reverses sign across network
   regions. On a CelebA-HQ 256 denoiser the timestep main effect dominates and no reversal
   occurs. Sensitivity structure is model-dependent.
3. A proxy-validity negative result. Allocating to minimise our paired metric improves that
   metric at matched bit-width but loses to the separable allocation on FID at the only
   budget where images are recognisable. We report this in the body rather than in a
   limitations section, because it bounds how any such proxy may be used.

We do not claim that separable methods are wrong in general, that the interaction is
universally large, or that the cross-term is generally worth optimising. One of our two
models supports none of those statements.

%% S2. Related work

**Per-layer sensitivity.** PTQ4DM [11] and Q-Diffusion [10] established post-training
quantization for diffusion models and set the calibration practice later work inherits, both
treating sensitivity as a per-layer property. Qua2SeDiMo [1] builds explainable sensitivity
estimates over denoiser weight layers, reaching 3.4–3.9 bit weight quantization on PixArt,
Hunyuan-DiT and SDXL, and MPQ-DM [5] allocates layer-wise precision from a kurtosis
criterion. MixDQ [13] allocates per layer under a metric-decoupled score, on the premise
that the sensitivity metric conventionally used for allocation does not track image quality;
Section VIII is an independent and sharper instance of that premise. In all of these the
denoising timestep enters through the calibration data rather than through the allocation.

**Per-timestep precision.** TDQ [6] makes quantizer parameters timestep-dependent at fixed
bit-width. TFMQ-DM [12] identifies the temporal feature pathway as the component whose
quantization error propagates along the trajectory and protects it specifically, which is
direct evidence that different parts of the network behave differently in time, though it
does not measure a joint grid. TMPDiff [2] varies the bit-width per timestep and is the
closest prior work to our temporal axis. It validates additive accumulation across timesteps
using the final-latent deviation, reporting r > 0.94, with one bit-width per timestep shared
by all layers, so it holds the layer axis fixed by construction.

**Joint plans and error accumulation.** DiffPro [3] tunes timesteps and per-layer precision
together for up to 6.25x compression, collapsing activations recorded across early, mid and
late timesteps to one number per layer; TMPQ-DM [4] pairs non-uniform timestep grouping with
layer-wise bit allocation. Both rely on a separable surrogate to make the search tractable,
so their tractability arguments depend on the assumption measured here. Separately,
AccuQuant [7] shows single-step calibration is insufficient, and timestep-aware correction
[8] and drift correction [9] compensate the accumulated perturbation. That literature treats
accumulation as something to correct; we treat the joint layer-time structure as something
to measure first.

%% FIG: fig/f1_probe.png
**Fig. 1.** The probe. **(a)** A reference trajectory holds every layer at full precision; the probe quantizes one layer group to b bits, and only while the sampler is inside one timestep band. Both trajectories are driven by the same noise draw, so the paired metric D of (5) carries no sampling variance. **(b)** The cells of the sensitivity grid, each probed in isolation.

%% S3. Method

**The probe.** Partition the denoiser's quantizable convolutions into layer groups G and
normalised timestep t/T into bands B. A cell is a pair (g,b), as in Fig. 1. To measure a
cell's marginal damage we quantize the convolutions in g to b bits only while the sampler is
inside band b, holding every other layer, and the same layers at every other timestep, at
full precision. Bit-width is selected at run time by a schedule evaluated against the current
timestep and broadcast through a module-level context set once per denoising step,

%% EQ 4

%% CONT
where g0 is the probed group and the band spans t/T in [tau0, tau1). Because nothing else
changes, the resulting quality drop is attributable to that cell. Every transformation
returns the number of modules it replaced and raises if that number is zero, and after
sampling each timestep-aware layer is checked to confirm it observed a timestep during the
run, so a transformation that silently reports no effect is treated as a failure rather than
a no-op.

**The metric.** We score cells by paired deviation from the full-precision trajectory rather
than by FID [14]. Each configuration is sampled from the same fixed noise stream as the
full-precision reference and scored by the paired per-pixel RMS difference between matched
images, over N images of P pixels each,

%% EQ 5

%% CONT
where x and x-hat are the reference and probed images generated from the same noise draw.
Common random numbers remove the sampling variance that dominates FID at the sample budget
available to us: across the allocation arms we evaluate, the seed-to-seed standard deviation
of FID ranges from 0.03 to 2.24, the same order as the damage of a weakly sensitive cell,
which would leave most of the grid unresolvable.

**Reproducibility of the instrument.** With cuDNN algorithm selection pinned and TF32
disabled, an identical re-draw of the reference differs by exactly zero, so under fixed
execution conditions the measurement carries no stochastic component. We claim
reproducibility under those conditions, not exactness in general. An earlier version of the
protocol compared re-draws at a different batch size; the resulting backend algorithm change
produced a spurious drift of 9.3e-4, the same order as a weakly sensitive cell's damage and
large enough to have corrupted the grid without failing visibly.

**The decomposition.** Let D[g,b] be measured damage at one bit-width. The separability
assumed by prior work is a per-layer term composed with a per-timestep term, which in log
space is the two-way main-effects model

%% EQ 6

%% CONT
with no interaction term, equivalently D[g,b] = A f(g) h(b). We fit it by row and column
means and split the total log-variance into a layer effect, a timestep effect and an
interaction residual,

%% EQ 7

%% CONT
and report each as a share of the total. The interaction share is the quantity of interest:
it is the part of the sensitivity surface that no separable table can represent, and Fig. 2
draws it for the CIFAR-10 DDPM at W1. We deliberately do not fit the best rank-one outer
product of the log matrix. That model carries a free per-group multiplier and can itself flip
the sign of the temporal trend, so it would conceal the effect under test. Main effects is
the stricter and the correct null.

%% FIG: fig/f2_grid.png
**Fig. 2.** The decomposition, with measured values from the CIFAR-10 DDPM at W1. **(a)** The measured damage grid D[g,b]. **(b)** The main-effects fit of (6), which is what a separable sensitivity table encodes. **(c)** The interaction residual, measured divided by fit: the separable table is wrong by a factor of 2.9 on the worst cell, and in opposite directions in different regions of the grid.

%% CONT

%% S4. Experimental setup

**Models.** A pixel-space DDPM [16] for CIFAR-10 at 32x32 (`google/ddpm-cifar10-32`), 63
quantizable convolutions grouped 11/9/4/39. A pixel-space denoiser for CelebA-HQ at 256x256
(`google/ddpm-celebahq-256`), 84 quantizable convolutions grouped 10/11/4/59, a different
dataset, eight times the resolution and a larger U-Net [18]. Both are sampled with DDIM [17]
over 50 steps, with the beta schedule taken from each model's own configuration. Input and
output projections stay at full precision. We quantize weights only; activations are not
quantized.

**Quantization.** Uniform per-output-channel symmetric weight quantization. For output
channel c with scale s_c, a weight w is mapped by

%% EQ 1

%% CONT
and the scale is chosen per channel to minimise weight MSE and cached once per layer per
bit-width,

%% EQ 2

%% CONT
solved by a sweep over candidate clipping ranges. One bit is the degenerate case, where the
grid has no interior and the optimal symmetric reconstruction is the sign pattern scaled by
the channel's mean magnitude,

%% EQ 3

%% CONT
There is no calibration-set reconstruction step of the kind PTQ4DM [11] and Q-Diffusion [10]
use, because the object under study is the sensitivity surface of the unmodified network
rather than the best achievable quantized model.

**Grids.** Four layer groups and four equal timestep bands. CIFAR-10: bit-widths {1,2,4,8},
64 cells, 256 paired images each. CelebA-HQ: bit-widths {1,4}, 32 cells, 64 paired images
each. All grid results are measured on an NVIDIA T4; a partial replication on an RTX 2050
appears in Section IX. FID [14] is computed with clean-fid [15] conventions over 2000
samples against the CIFAR-10 training set, three noise streams per arm.

%% S5. Regime one: a CIFAR-10 DDPM

Table I gives the ANOVA decomposition. The layer axis dominates, as prior work assumes. The
finding is the relative size of the two remaining terms: the interaction exceeds the timestep
main effect by 4.4 to 6.4 times, at every bit-width measured.

%% TABLE1
CIFAR-10 DDPM. Two-way ANOVA on log damage, as a share of total log-variance. The last
column is the largest factor by which the main-effects fit misses a single cell.
Bit-width | Layer | Timestep | Interaction | Worst cell miss
W1 | 86.9% | 2.0% | 11.2% | x2.93
W2 | 85.3% | 2.7% | 11.9% | x4.42
W4 | 89.7% | 1.7% | 8.6% | x3.21
W8 | 81.6% | 2.5% | 16.0% | x4.79

%% CONT
The mechanism is a sign reversal (Table II). Two of the four groups worsen toward the noise
end of the trajectory and two toward the detail end, so a timestep score averaged over layers
nearly cancels. No sum of a per-layer term and a per-timestep term can change sign between
layers. The separable fit mispredicts individual cells by up to a factor of 4.8. Allocation
decisions are made per cell, so that is the error that matters rather than aggregate fit
quality.

%% TABLE2
CIFAR-10 DDPM at W1. The temporal gradient reverses sign between layer groups. Damage is
paired per-pixel RMS.
Layer group | Detail band | Noise band | Ratio | Direction
down 0-1 | 0.0501 | 0.2861 | 5.71 | worsens toward noise
up | 0.2600 | 0.4368 | 1.68 | worsens toward noise
down 2-3 | 0.0157 | 0.0036 | 0.23 | worsens toward detail
mid | 0.0082 | 0.0025 | 0.30 | worsens toward detail

%% S6. Regime two: a CelebA-HQ denoiser

Repeating the identical protocol on CelebA-HQ 256 inverts the picture (Table III). The
timestep main effect is now the large term at 37.9–51.0%, the interaction falls to 4.0–10.1%,
and the worst cell miss of the separable fit drops to 1.43x at W1. No sign reversal occurs:
all four groups worsen toward the noise end, by factors of 2.2 to 6.2 at W1 and 2.3 to 22.3
at W4. Because the temporal trends share a sign they reinforce rather than cancel under
marginalisation, and a per-timestep summary carries real information.

%% TABLE3
CelebA-HQ 256. The same decomposition, with the ordering of the timestep and interaction
terms inverted.
Bit-width | Layer | Timestep | Interaction | Worst miss | Rising
W1 | 58.1% | 37.9% | 4.0% | x1.43 | 4/4
W4 | 39.0% | 51.0% | 10.1% | x2.52 | 4/4

%% CONT
The contrast between Tables I and III is the paper's central result. On one model the
timestep axis carries almost no separable signal and the interaction dominates it; on another
the timestep axis dominates and the interaction is secondary. A separable sensitivity table
is close to adequate for the second model and materially wrong for the first.

The two models differ in dataset, resolution and network size at once, so we cannot attribute
the difference to any one factor and we do not try to. One alternative explanation deserves
naming. TFMQ-DM [12] localises temporal quantization damage to the time-embedding pathway.
Our layer partition covers only down, mid and up convolutional blocks, so if the timestep
main effect lives mainly in the embedding path, the small CIFAR timestep share may partly
reflect layers that were never in the grid. The claim we make is the weaker and
better-supported one: the structure is model-dependent and must be measured.

%% FIG: fig/f3_regimes.png
**Fig. 3.** Sensitivity structure is model-dependent. **(a)** On the CIFAR-10 DDPM the temporal profiles of the four layer groups cross, so a timestep score marginalised over layers nearly cancels. **(b)** On the CelebA-HQ 256 denoiser, measured with the identical protocol and drawn on the same logarithmic axis, all four profiles share a sign. **(c)** The two-way ANOVA split of (7) for both models at every bit-width measured, with the interaction share printed at the right.

%% S7. Compositional prediction

Sensitivity tables are only useful if the damage of a multi-cell configuration follows from
its parts. TMPDiff [2] validates this along the time axis with layers uniform, reporting
r > 0.94. We repeat the test on CIFAR-10 with both axes free: 32 configurations, each
quantizing a random subset of two to eight cells at a random bit-width, measured against the
sum of the corresponding single-cell damages.

Prediction and measurement remain correlated (r = 0.888 in linear space) but the relation is
not the identity. Measured damage averages 0.72 of the additive prediction, and the deficit
is systematic rather than noise. A multiplicative fit over the K cells of a configuration,

%% EQ 8

%% CONT
gives an exponent of 0.93 +/- 0.03 with R-squared 0.97, against R-squared 0.79 for the
ordinary least-squares fit in linear space, whose slope of 0.53 carries an intercept of 0.044
that has no interpretation at zero damage. We prefer the multiplicative fit: an exponent
below one is exactly the statement that damage accumulates sub-additively, and unlike the
linear fit it passes through the origin.

Damage is therefore sub-additive once both axes vary, and the deficit grows both with the
predicted damage of the configuration (Spearman rho = -0.51, p = 0.003) and with the number
of cells quantized together (rho = -0.49, p = 0.005); the two predictors are correlated in
our design, so we do not separate them. The consequence is the same either way: an allocator
that sums single-cell costs over-estimates the damage of aggressive configurations and is
therefore too conservative rather than too aggressive.

%% S8. The proxy is not the objective

Sections V to VII measure structure in a trajectory-deviation metric. This section tests
whether minimising that metric improves generation quality, and finds that it does not
reliably do so. We fix an average-bit budget B and solve exactly, by dynamic programming over
a discretised budget, for the per-cell assignment minimising total predicted damage,

%% EQ 9

%% CONT
where b_gb is the bit-width assigned to cell (g,b). Two allocators are compared, differing
only in whether the damage table D is the measured grid or its main-effects fit from (6).
Scored by the paired metric they were built to minimise, the measured allocator wins at every
budget, by 28.9, 4.1, 47.9 and 12.9% at 2, 3, 4 and 6 average bits, a mean of 23.4%. The
per-budget gain correlates with the number of cells on which the two plans disagree at
r = 0.968, so the benefit arises where the separable approximation misleads the allocator
rather than anywhere else. The two plans are never identical, and the measured allocator
finishes marginally under budget at every setting, so it holds no bit-width advantage over
the arm it beats.

Re-scored with FID over three noise streams, the ordering does not survive (Table IV).

%% FIG: fig/f5_proxy.png
**Fig. 4.** The proxy disagrees with the objective where it matters. **(a)** The paired trajectory metric the allocators minimise; the measured allocator wins at every budget. **(b)** FID for the same plans. The shaded band marks the region where no arm produces recognisable images, and the dashed line is the FP32 baseline. **(c)** The two metrics in an agreement plane, each point one budget; both axes are the percentage by which the measured allocation improves on the separable one. The only budget with a meaningful FID comparison, B = 6, sits deep in the shaded half-plane where the proxy is improved and FID is made worse.

%% TABLE4
CIFAR-10 FID, mean over three noise streams, 2000 samples, against an FP32 baseline of
28.44 +/- 0.25. Achieved average bits in parentheses. A uniform arm cannot sit at 6 average
bits with bit-widths {1,2,4,8}, so both bracketing widths are reported there.
Budget | Uniform | Separable | Measured
2.0 | 386.7 (2.00) | 385.0 (1.99) | 354.3 (1.98)
4.0 | 385.1 (4.00) | 406.2 (4.00) | 269.0 (3.98)
6.0 | 385.1 (W4) / 30.7 (W8) | 43.3 (5.98) | 98.2 (5.97)

%% CONT
Three observations. At 2 and 4 average bits the measured allocator is ahead, but every arm
sits at FID 269–406 against a full-precision baseline of 28.4. These configurations do not
produce recognisable images, and an ordering among them is not a perceptual result. At 6
average bits, where the separable allocation reaches FID 43.3 and images are recognisable,
the measured allocation is substantially worse at 98.2. The proxy and the generation
objective disagree exactly where the comparison begins to mean something.

The uniform bracket at 6 average bits also bounds what allocation buys at all. Uniform W8
reaches FID 30.7 +/- 0.9, close to the full-precision 28.4, while uniform W4 collapses to
385.1. Both allocations sit between those brackets while spending about 6 bits: the separable
one converts a two-bit saving over W8 into a 12.6-point FID cost, and the measured one into a
67.5-point cost. Neither beats simply using eight bits, and we do not claim otherwise; what
the comparison isolates is which of two allocations at the same budget is better, and there
the proxy chooses wrongly.

We therefore withdraw any claim that the cross-term is generally worth optimising. What
survives is narrower and more useful. A trajectory-deviation metric can be extremely
reproducible and highly sensitive as an instrument for measuring local interventions while
being an inadequate objective for allocation. Reproducibility of a proxy is not evidence of
its validity, and sensitivity proxies should be validated against the end-task metric before
they are optimised.

Two mechanisms could produce this gap and our data does not separate them. The paired RMS
metric weights all pixels and timesteps equally, whereas FID responds to distributional
statistics of the final image, so an allocation that reduces mean trajectory deviation may
concentrate what remains in the late, detail-forming steps FID is most sensitive to.
Alternatively the sub-additivity of Section VII may bite unevenly. Re-weighting the damage
table toward late bands and re-solving distinguishes these, and is the natural next
experiment.

%% S9. Validity

**Measurement floor.** With cuDNN algorithm selection pinned, an identical re-draw of the
reference differs by exactly zero, so under fixed execution conditions the paired comparison
carries no stochastic component. The CIFAR grid spans a factor of 2.1e4 between its weakest
and strongest cell.

**Seeds and sessions.** Allocation results are reported over three independent noise streams,
with a per-budget standard deviation of 0.12–0.52 percentage points; two independent sessions
on separate machine allocations returned bit-identical results over all 64 cells.

**Hardware.** A partial replication on an RTX 2050 agrees with the T4 at Pearson r = 0.9991
on log damage and rank correlation 0.9985 over 47 overlapping cells, median absolute relative
difference 4.9%. Disagreements concentrate where absolute damage is of order 1e-4; values
below roughly 1e-4 should not be compared across devices. The structure we report is
unaffected.

%% S10. Limitations

**Two models is not a general claim.** We observe two regimes and conclude only that the
structure varies. Establishing what governs it, whether architecture family, latent versus
pixel space, resolution or dataset, needs a designed sweep with those factors separated.

**Weights only.** We quantize weights, not activations, so the temporal dependence we measure
enters solely through activations flowing through fixed perturbed weights. Methods that
quantize activations may see larger timestep effects, and the TFMQ-DM [12] result on temporal
features suggests where they would appear.

**Unequal budgets and coarse grouping.** The CelebA-HQ grid uses 64 paired images per cell
and two bit-widths against CIFAR's 256 and four, so the two grids are not matched in
statistical power, though the effect sizes there are large. Four layer groups pool 39 of 63
convolutions into a single `up` group on CIFAR and 59 of 84 on CelebA-HQ, and a pilot found
two blocks inside the CIFAR `up` group differing by 102x at W1 in the noise band, so the
interaction in Table I is best read as a lower bound: finer grouping can only move variance
out of the pooled residual and into resolved cells.

**FID budget and pipeline.** FID is computed from 2000 samples where published protocols use
50k, and our quantizer has no reconstruction step, so absolute values are not comparable to
published numbers. The gap at 6 average bits, 43.3 against 98.2, is far larger than the seed
spread we observe, which is what the argument of Section VIII rests on.

%% S11. Conclusion

We tested whether layer and timestep sensitivity separate in diffusion quantization and found
that the answer depends on the model. A CIFAR-10 DDPM shows a near-vanishing timestep main
effect and a dominant interaction driven by a sign reversal across network regions; a
CelebA-HQ denoiser shows the reverse. Neither a globally separable sensitivity model nor a
single trajectory-level allocation heuristic can be assumed valid without measurement, and
the protocol given here makes that measurement cheap and exactly reproducible within a
device.

We also aimed to show that the interaction term is worth optimising, and that claim did not
survive its own validation: the allocation minimising our paired trajectory metric is beaten
on FID by the separable allocation at the only budget producing recognisable images. We report
it because the failure is informative. A proxy can be exactly reproducible and still be the
wrong objective, and a measurement programme that does not check its proxy against the
generation metric is not measuring what it believes it is measuring.

%% S12. Data and code availability

Every number in this paper is released as a tidy CSV, one row per measured observation with
no aggregation applied, together with the scripts that produce the figures and tables from
them.

%% ACKNOWLEDGMENT
The authors used Anthropic's Claude to assist with implementing the measurement harness and
analysis scripts described in Sections III and IV, with generating the figures, and with
drafting and editing portions of the manuscript. All experiments were designed, executed and
verified by the authors, who take full responsibility for the content and the conclusions of
this work.

%% REFERENCES
[1] K. G. Mills, M. Salameh, R. Chen, N. Hassanpour, W.-Y. Lu, and D. Niu, "Qua2SeDiMo: Quantifiable quantization sensitivity of diffusion models," Proc. AAAI Conf. Artificial Intelligence, vol. 39, no. 6, 2025, doi:10.1609/aaai.v39i6.32658.
[2] B. Lewandowski, S. Kurz, A. Shankar, R. Birke, J.-J. Chen, and L. Y. Chen, "TMPDiff: Temporal mixed-precision for diffusion models," arXiv:2603.14062, 2026.
[3] F. Amin, S. Afroz, K. Gharami, M. Moghadampanah, and D. S. Nikolopoulos, "DiffPro: Joint timestep and layer-wise precision optimization for efficient diffusion inference," Lecture Notes in Computer Science, Springer, 2026, doi:10.1007/978-3-032-37252-9_7.
[4] H. Sun, C. Tang, Z. Wang, Y. Meng, J. Jiang, X. Ma, and W. Zhu, "TMPQ-DM: Joint timestep reduction and quantization precision selection for efficient diffusion models," arXiv:2404.09532, 2024.
[5] W. Feng, H. Qin, C. Yang, Z. An, L. Huang, B. Diao, F. Wang, and R. Tao, "MPQ-DM: Mixed precision quantization for extremely low bit diffusion models," Proc. AAAI Conf. Artificial Intelligence, 2025, doi:10.1609/aaai.v39i16.33823.
[6] J. So, J. Lee, D. Ahn, H. Kim, and E. Park, "Temporal dynamic quantization for diffusion models," Advances in Neural Information Processing Systems, 2023.
[7] S. Lee, J.-W. Choi, B. Son, J. Moon, J. Jeon, and B. Ham, "AccuQuant: Simulating multiple denoising steps for quantizing diffusion models," Advances in Neural Information Processing Systems, 2025, doi:10.52202/085713-3679.
[8] Y. Yao, F. Tian, J. Chen, H. Lin, G. Dai, Y. Liu, and J. Wang, "Timestep-aware correction for quantized diffusion models," Proc. European Conf. Computer Vision (ECCV), pp. 215-232, 2024, doi:10.1007/978-3-031-72848-8_13.
[9] S. Ryu, M. Salzmann, and S. Javed, "Q-Drift: Quantization-aware drift correction for diffusion model sampling," arXiv:2603.18095, 2026.
[10] X. Li, Y. Liu, L. Lian, H. Yang, Z. Dong, D. Kang, S. Zhang, and K. Keutzer, "Q-Diffusion: Quantizing diffusion models," Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2023, doi:10.1109/ICCV51070.2023.01608.
[11] Y. Shang, Z. Yuan, B. Xie, B. Wu, and Y. Yan, "Post-training quantization on diffusion models," Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2023, doi:10.1109/CVPR52729.2023.00196.
[12] Y. Huang, R. Gong, J. Liu, T. Chen, and X. Liu, "TFMQ-DM: Temporal feature maintenance quantization for diffusion models," Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2024, doi:10.1109/CVPR52733.2024.00703.
[13] T. Zhao, X. Ning, T. Fang, E. Liu, G. Huang, Z. Lin, S. Yan, and G. Dai, "MixDQ: Memory-efficient few-step text-to-image diffusion models with metric-decoupled mixed precision quantization," Proc. European Conf. Computer Vision (ECCV), 2024, doi:10.1007/978-3-031-72630-9_17.
[14] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," Advances in Neural Information Processing Systems, 2017.
[15] G. Parmar, R. Zhang, and J.-Y. Zhu, "On aliased resizing and surprising subtleties in GAN evaluation," Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2022, doi:10.1109/CVPR52688.2022.01112.
[16] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," Advances in Neural Information Processing Systems, 2020.
[17] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," Int. Conf. Learning Representations (ICLR), 2021.
[18] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," Proc. MICCAI, pp. 234-241, 2015, doi:10.1007/978-3-319-24574-4_28.
