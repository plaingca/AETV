# Cross-GOP temporal continuity for AETV V8

**Repository baseline:** `origin/main` / `main` commit `91f9a6d` (2026-08-26)

**Review date:** 2026-08-26

**Scope:** excess temporal inconsistency introduced where independently encoded one-second V8 GOPs meet

## Executive summary

1. **V8 never trains on the event that fails.** The main trainer requests exactly six frames, performs one encoder/decoder pass, and computes every temporal objective inside that tensor. It supervises transitions 0→1 through 4→5, but never 5→6 after two independent passes. The production codec is likewise stateless across `encode_gop` and `decode_gop` calls. This is the strongest code-grounded cause, and two-GOP boundary-aware training is the lowest-risk V8.1 experiment.

2. **V8 already has extensive ordinary temporal supervision.** It has signed frame-delta L1, temporal acceleration, motion-energy and cosine objectives, temporal perceptual loss, 3D Haar/DWT loss, 3D PatchGAN, axial temporal attention, and a direct temporal latent-to-RGB skip. Recommending another generic temporal loss misses the gap. The missing extension is to form the reconstructed transition using two independent codec calls and explicitly score that transition.

3. **Direct literature supports overlap, but literal frame overlap is rate-expensive for V8.** *Perceptual Neural Video Compression with Video Variational Autoencoder at Low Bitrates* explicitly describes discontinuities between independently processed GOPs and reconstructs a shared transition frame twice. Free-GVC similarly overlaps frames and fuses latent representations. With six-frame V8 windows, one-frame overlap changes the rate from `2816/6` to `2816/5` values per source frame: **+20% symbols/s**. A context-only variant that consumes previous decoded frames but transmits only the six new frames is therefore more attractive.

4. **The best zero-RF-bit architecture is short, resettable, receiver-reliability-gated context.** Feed the preceding reconstruction or aligned decoder feature into a small residual adapter for only the first frames of the new GOP. Gate it with the previous/current latent confidence, train it under clean/AWGN/fading state mismatch, taper its correction to zero, and bypass exactly after loss or reacquisition. This avoids an indefinite predictive chain.

5. **The most publishable gap is narrower than “temporal memory for video.”** Prior learned codecs propagate decoded features; recent JSCC work studies asymmetric TX/RX context and multipath OFDM; recent codecs suppress unreliable temporal references. I did not find a paper combining **analog/pseudo-analog video JSCC, independently framed GOP recovery, decoder-synchronous cross-GOP state, per-latent radio reliability, and learned confidence/reset control under fading**. That combination is a credible research contribution if evaluated on boundary-specific metrics and fade recovery rather than average PSNR alone.

## 1. Current AETV V8 architecture

### 1.1 Radio and runtime contract

V8 is `192×108` RGB at `6 fps`, with six video frames in exactly one RF second. It uses the W-band OFDM geometry: 45 carriers at 50 Hz spacing, 44 payload carriers, 32 data symbols per GOP, and I/Q payload mapping for **2,816 real values/GOP**. Carrier centers span 450–2650 Hz and TX conditioning spans 350–2750 Hz. It is continuous-valued analog/JSCC-style transport, not an entropy-coded file codec.

The requested `aetv/attention.py` path does not exist at current main; `AxisAttention3d` is defined in `aetv/video_backbone.py`. The similarly named `docs/v8-ota-perceptual.md` and `docs/v8-ota-rxfix.md` describe the older Flex-8k/V7 waveform-contract experiment, not the HF-3k V8 mode. They are relevant evidence for confidence-aware noisy-latent decoding, but `docs/v8-hf3k.md`, the installed checkpoint arguments, and the V8 entry in `aetv/config.py` are authoritative for the architecture characterized here.

The runtime path is:

`six source frames → AETVCodec.encode_gop → 2816 real unit-RMS values → OFDM/interleaver/channel → 2816 equalized values + 2816 confidence weights → AETVCodec.decode_gop → six reconstructed frames`

`AETVCodec.encode_gop()` and `decode_gop()` allocate all inputs locally and return NumPy arrays. They do not accept or return recurrent state. The live station calls them once per recovered GOP. The streaming modem retains synchronization/drift/channel-tracking machinery, but **the neural image encoder and decoder retain no previous frame, feature, latent, attention cache, motion, appearance, or reliability state across GOPs**.

The GUI separately calls `blend_gop_boundary(previous, frames, transition_frames=4)`. It adds 80% of the previous-to-current first-frame RGB offset to the first new frame and tapers that offset to zero over four frames. This is display-only concealment: it is not learned, not source-motion-aware, not used by raw codec evaluation, and can hide a reconstruction offset by spreading it over most of a six-frame GOP.

### 1.2 Encoder

The released `models/v8-hf3k-perceptual.pt` checkpoint records `model_width=128`, `latent_channels=3`, and `compact=false`. V8 is noncausal (`mode.causal=false`). Its main encoder is therefore:

- 3D convolution `3→64`, kernel `3×5×5`, stride `1×2×2`;
- deep 3D residual stack at half spatial resolution;
- 3D convolution `64→128`, kernel `3³`, stride `2×2×2`;
- deeper residual stack, plus spatial-then-temporal axial attention at quarter resolution;
- 3D convolution `128→256`, kernel `3³`, stride `1×2×2`;
- deep residual stack and bottleneck axial attention;
- 3D projection `256→3` and `tanh`.

All checkpoint convolutions are temporally symmetric because V8 is noncausal. Axial attention first attends over every spatial location within each latent time slice and then over all three latent time slices at every spatial position. Consequently, the effective temporal receptive field covers the complete six-frame GOP; it stops absolutely at the GOP boundary.

The encoder grid is `3 channels × 3 time × 14 height × 24 width = 3024` values. It is flattened and prefix-truncated to 2,816 transmitted coordinates. The backbone performs whole-grid RMS normalization (`clip_rms_latents=true` by default), and `AETVEncoder` again normalizes the transmitted 2,816-value vector to unit RMS. GroupNorm is used throughout the released construction.

Compact mode is present in the source but is **not** the released V8 layout. Compact mode keeps six latent time slices and downsamples space by 16; it should not be used to describe the installed checkpoint.

### 1.3 Decoder

For noncompact V8, the decoder reconstructs a `3×3×13×24 = 2808` latent grid. It copies the first 2,808 received values; therefore the last eight of the 2,816 radio values do not enter the current decoder grid. It forms both:

- `z × confidence`, and
- the confidence grid itself.

These are concatenated at the input, so the decoder can distinguish a reliable zero from an erased/unreliable coordinate. A separate `1×1×1` temporal skip maps `z × confidence` directly to RGB logits and trilinearly upsamples it to six output frames.

The synthesis trunk has residual blocks and factorized spatial/temporal attention at the bottleneck, three 2× spatial upsampling stages, nearest temporal interpolation from three to six slices at quarter resolution, additional attention at quarter resolution, and deep residual synthesis at half/full resolution. The optional half-resolution `deep4` attention exists in source but is disabled by the default V8-size construction. The final output is `sigmoid(main_logits + temporal_skip)`.

The decoder is confidence-aware **within the current GOP**, but neither the confidence nor any decoded feature survives into the next GOP. Missing/damaged coordinates are attenuated by their weights, while the explicit weight plane tells the decoder where information is weak. This is the natural control signal for a future memory gate.

### 1.4 What training already does

The main trainer includes:

| Mechanism | Main V8 status | Boundary limitation |
|---|---|---|
| MSE and pixel L1 | Present | One GOP only |
| Spatial gradient L1 | Present | No cross-pass transition |
| Signed temporal-delta L1 | Present; released checkpoint weight 5.0 | Scores only five within-GOP deltas |
| Temporal acceleration L1 | Present; weight 2.0 | Scores only within-GOP triplets |
| Per-frame motion-energy match | Present; weight 10.0 | No boundary energy |
| Signed temporal cosine | Present; weight 0.2 | Flattens only the current GOP's delta tensor |
| Multi-layer VGG perceptual | Present; weight 0.1 | Frame-wise |
| Perceptual loss on signed deltas | Present; weight 0.1 | Only five within-GOP deltas |
| 3D Haar/DWT | Present; weight 1.0 | Transform never spans two codec calls |
| 3D local PatchGAN and feature matching | Implemented; disabled in released checkpoint (`adv=0`, `fm=0`) | A discriminator invocation sees one GOP |
| Axial spatial/temporal attention | Present | Attention scope resets each GOP |
| Direct temporal skip | Present | Interpolates only the current latent grid |
| Channel-corrupted render training | Present in Stage 2 | Corrupts a single GOP |
| Clean/noisy reconstruction consistency | Present | Not TX/RX state consistency |
| Long clips / explicit independent boundary | **Absent from main trainer** | Central missing supervision |
| Optical-flow-warped boundary loss | Absent | Candidate extension |
| Persistent/recurrent state | Absent | Candidate extension |
| Boundary-specific discriminator | Absent | Candidate extension |

The dataset queue constructs `VideoClipSpec(frames=mode_spec.gop_frames)`, which is six for V8. The training step encodes and decodes this tensor once. Thus the hypothesis in the request is confirmed exactly: training covers 0→1, 1→2, 2→3, 3→4, and 4→5, but never 5→6 where each side was produced by a separate call.

### 1.5 Main versus post-main worktree experiments

The checkout also contains uncommitted experimental artifacts that are not part of `origin/main` and are not enabled in the GUI. They are useful evidence, not shipped architecture:

- A resettable 101,547-parameter RGB context adapter reportedly reduced all measured boundary errors over paired 32-sequence clean/AWGN6/MPP12 cells, with 0.36 ms/GOP GPU cost and a 0.4–0.6% within-GOP delta regression.
- A 1.55M-parameter RAFT-aligned decoder-feature refiner plus 10% output fusion reduced boundary delta by 11.7–13.2%, low-pass boundary step by 22.1–24.5%, and boundary-delta LPIPS by 9.2–10.7%, with flat/improved overall LPIPS and about 12.2 ms/boundary on an RTX 4090.
- Faithful one-frame overlap reduced seam metrics but required +20% symbol rate; a rate-neutral truncation variant failed spatial-quality gates.

These local results strengthen the ranking below, but the literature comparison labels “Already in V8?” against released/main V8 unless stated otherwise.

## 2. Direct literature on GOP/chunk/reset boundaries

### Anonymous ICLR 2026 submission: frame overlap

[*Perceptual Neural Video Compression with Video Variational Autoencoder at Low Bitrates*](https://openreview.net/forum?id=nfOnCngWtp), anonymous authors, 2026, ICLR submission/preprint.

This is the clearest direct match. The paper says independent GOP processing does not explicitly model adjacent-group correlation and can cause visible discontinuities, exacerbated by quantization. Adjacent `T`-frame GOPs share one source frame. The decoder reconstructs that frame twice and replaces both versions by `θ x̂_left + (1-θ) x̂_right`. It requires no retraining. The paper's `T=9` asymptotic relative overhead is 12.5%; for V8 `T=6`, it is 20%. Applicability: **Direct**. It addresses a transition sample rather than the different decoder interpretation of all early frames, and it spends rate that V8 does not have.

### Free-GVC: overlapping latent fusion

[*Free-GVC: Towards Training-Free Extreme Generative Video Compression with Temporal Coherence*](https://arxiv.org/abs/2602.09868), Xiaoyue Ling, Chuqin Zhou, Chunyi Li, Yunuo Chen, Yuan Tian, Guo Lu, Wenjun Zhang, 2026, arXiv preprint.

Free-GVC identifies independent group processing as a source of inter-GOP temporal discontinuity and flicker. It overlaps `m` source frames, encodes both copies, and gradually blends the corresponding compressed latent features before video-VAE decoding. Its example (`l=48`, `m=4`, two GOPs) adds under 4.4% rate. Literal application with V8 `l=6,m=1` again adds 20% steady-state RF rate. Applicability: **Direct**. Latent fusion is more architecturally relevant than RGB averaging, but Free-GVC's diffusion trajectory coding is far too expensive and structurally different for live AETV.

### StreamingT2V: independent chunks become hard cuts

[*StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text*](https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html), Roberto Henschel et al., 2025, CVPR.

Naive chunk extension produces hard cuts. StreamingT2V combines short-term cross-chunk conditional attention, a long-term appearance module anchored to the first chunk, and overlap with shared noise/randomized latent blending. The enhancer uses 24-frame chunks with eight-frame overlap. Applicability: **Strongly applicable mechanism, speculative implementation**. The key transferable lesson is that local transition memory and slowly changing appearance memory solve different failure modes. Its diffusion compute and 33% overlap are incompatible with V8.

### Consistent context-aware sliding inference

[*Learning Temporally Consistent Video Depth from Video Diffusion Priors*](https://openaccess.thecvf.com/content/CVPR2025/papers/Shao_Learning_Temporally_Consistent_Video_Depth_from_Video_Diffusion_Priors_CVPR_2025_paper.pdf), Jiahao Shao et al., 2025, CVPR.

The paper shows that naive independent sliding windows flicker and proposes reusing previously predicted overlap frames as clean context rather than renoising them. Applicability: **Strongly applicable** to zero-bit previous-reconstruction context, but the task and diffusion backbone differ. Its warning is important for AETV: corrupting/re-randomizing context differently on two sides recreates state mismatch.

## 3. Long-term temporal context in learned video codecs

### Learned arbitrary state

[*Learned Video Compression*](https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html), Oren Rippel, Sanjay Nair, Carissa Lew, Steve Branson, Alexander G. Anderson, Lubomir Bourdev, 2019, ICCV.

Rippel et al. maintain learned state rather than relying only on previously transmitted reference frames, enabling general learned compensation. Applicability: **Strongly applicable foundation**. It establishes zero-extra-bit decoder state, but targets sequential low-delay RD and does not isolate reset seams or fading-state poisoning.

### ConvLSTM temporal prior

[*Learned Video Compression via Joint Spatial-Temporal Correlation Exploration*](https://arxiv.org/abs/1912.06348), Haojie Liu, Han Shen, Lichao Huang, Ming Lu, Tong Chen, Zhan Ma, 2020, AAAI/preprint.

The entropy prior recurrently incorporates temporal neighbors with ConvLSTM and uses first-/second-order motion. Applicability: **Strongly applicable architecture**, but it primarily saves bits through probability modeling. AETV has no entropy model, so the useful analogue is the recurrent feature state, not the spatial-temporal probability prior.

### Continuously updated GOP prior

[*Exploring Long- and Short-Range Temporal Information for Learned Video Compression*](https://arxiv.org/abs/2208.03754), Huairui Wang, Zhenzhong Chen, 2022, IEEE TMM/preprint (LSTVC).

LSTVC initializes a `64×H/4×W/4` prior from the I-frame and continuously updates/alines it with decoded frames and motion within a GOP. It reports about 12% additional bit saving from the temporal prior in one ablation. Applicability: **Partial analogue / adaptable**. It demonstrates dynamic retention/forgetting, but resets at the I-frame and measures RD, not continuity across the reset. For AETV the prior must cross the GOP boundary, be far smaller, and degrade safely.

### Diverse and high-quality reference contexts

[*Neural Video Compression with Diverse Contexts*](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.html), Jiahao Li, Bin Li, Yan Lu, 2023, CVPR (DCVC-DC).

DCVC-DC trains a hierarchical frame-quality pattern so high-quality long-term contexts are propagated, and uses offset diversity for motion compensation. Applicability: **Strongly applicable principle, incompatible full codec**. AETV could preserve a stable low-frequency/appearance context and use a small alignment module, but DCVC's entropy-coded conditional P-frame design and explicit motion machinery are a Class IV redesign.

### Asymmetric TX/RX context in pseudo-analog JSCC

[*Deep Joint Source-Channel Coding for Wireless Video Transmission with Asymmetric Context*](https://arxiv.org/abs/2601.06170), Xuechen Chen, Junting Li, Chuang Chen, Hairong Lin, Yishen Li, 2026, arXiv preprint.

This paper directly identifies the problem that a pseudo-analog JSCC encoder cannot know the decoder's exact reconstructed reference, even with simulated transmission. It lets TX and RX learn conditions from asymmetric contexts and propagates intermediate features independently on each side; the goal is to exploit long-range correlation while reducing error accumulation. It evaluates AWGN and Rayleigh fading and shows longer-GOP degradation in prior DVST. Applicability: **Strongly applicable and closest state-mismatch work**. It does not use AETV-style per-latent demodulator confidence to gate/reset state, and it transmits motion information.

### Robust video JSCC over multipath OFDM

[*Robust Deep Joint Source-Channel Coding for Video Transmission over Multipath Fading Channel*](https://arxiv.org/abs/2601.01729), Bohuai Xiao, Jian Zou, Fanyang Meng, Wei Liu, Yongsheng Liang, 2026, arXiv preprint.

The system combines OFDM, conditional multi-scale Gaussian-warped temporal features, and a lightweight denoiser, reporting 5.13 dB average reconstruction gain under its tested multipath conditions. Applicability: **Strongly applicable channel context, fundamental-redesign codec**. Its reported model uses 505.4 GFLOPs and does not isolate GOP seam metrics, so it is not evidence that it solves V8's boundary event.

### DeepWiVe

[*DeepWiVe: Deep-Learning-Aided Wireless Video Transmission*](https://arxiv.org/abs/2111.13034), Tze-Yang Tung, Deniz Gündüz, 2022, IEEE JSAC.

DeepWiVe maps a GOP directly to channel symbols, uses learned residual prediction without distortion feedback, dynamically allocates per-frame channel bandwidth, and degrades gracefully under channel mismatch. Applicability: **Strongly applicable JSCC baseline**. It is GOP-level and radio-aware, but its evaluation optimizes average MS-SSIM and does not propagate reliability-gated state across independently transmitted GOPs.

## 4. Persistent/recurrent state and error recovery

State should be separated by propagation horizon:

- **Boundary-local residual state:** previous final frame or aligned feature affects only frames 0–`k-1`, with correction forced to zero by frame `k`. Error propagation is strictly bounded to one GOP. This is the best V8 fit.
- **Leaky recurrent state:** `s_next = α s_old + (1-α) update`, with `α` learned from reliability and age. It can preserve appearance longer but needs explicit half-life, reset, and corruption tests.
- **Unbounded predictive state:** every reconstruction depends on all prior state. This gives the greatest coding gain and worst fade poisoning; it conflicts with AETV's independent reacquisition advantage.

The TX/RX synchronization choices are:

1. **Clean-TX state:** TX derives context from the clean source/local clean reconstruction, RX from its impaired reconstruction. This is cheap but mismatched by construction and should be a negative-control ablation, not the default.
2. **Closed-loop sampled state:** during training, TX runs the simulated channel/receiver and conditions on the same sampled reconstruction. This synchronizes a simulated pair, but a real receiver sees a different channel realization unless TX has feedback.
3. **Asymmetric learned state:** TX uses clean source context; RX uses noisy reconstructed context; training forces their resulting conditioning features to be compatible. This follows the 2026 asymmetric-context paper and requires no feedback.
4. **RX-only post-decoder adapter:** leave the encoder independent and use only previous RX reconstruction/features to correct the next decoded GOP. TX needs no state, eliminating TX/RX recurrence mismatch. The cost is that the base latent was not optimized to cooperate with context. This is the safest first architecture.

A robust update should consume a compact confidence summary and spatial confidence feature, not merely mean SNR. Candidate inputs are mean/min/quantiles of the 2,816 weights, the decoder weight grid, pilot SNR, dropout/gap flags, and novelty/scene-cut scores. Required hard safety behavior:

- exact bypass with no previous GOP or zero confidence;
- state clear on acquisition, missing GOP, mode/checkpoint/callsign change, ring overrun, or explicit discontinuity;
- learned or deterministic decay with bounded half-life;
- a scene-cut gate trained not to smooth true cuts;
- periodic random resets during training;
- sequences containing good→fade→good and good→drop→good cells;
- report recovery in GOPs after the impairment ends.

[*Uni-LVC*](https://arxiv.org/abs/2603.05756) (Yichi Zhang, Ruoyu Yang, Fengqing Zhu, 2026 preprint) is a close reliability precedent: it classifies unreliable temporal references and scales temporal cues so the codec approaches intra behavior. [*Uncertainty-Aware Deep Video Compression with Ensembles*](https://arxiv.org/abs/2403.19158) (Wufei Ma et al., 2024 preprint) models uncertainty in motion/quantized intermediates. Neither uses radio-demodulator confidence or boundary-local resettable memory.

## 5. Temporal consistency objectives

### What adds beyond V8's current losses

The necessary unit is a **pair of independently reconstructed GOPs**:

`ŷA = D(C(E(x0…x5)))`

`ŷB = D(C(E(x6…x11)))`

Then form source and reconstruction boundary deltas:

`R = x6 - x5`

`R̂ = ŷB[0] - ŷA[5]`

Useful objectives are:

- `L_boundary_delta = |R̂ - R|₁`;
- low-pass/luminance/chroma variants to target brightness and color jumps;
- spatial-gradient delta loss to target edge/texture-phase changes;
- perceptual distance between mapped signed deltas;
- second-order boundary acceleration using `ŷA[4], ŷA[5], ŷB[0]` and `ŷA[5], ŷB[0], ŷB[1]`;
- motion-compensated feature loss using source-derived flow and occlusion masks;
- latent/feature bias consistency after compensating real motion;
- a boundary discriminator presented with `[last k frames of ŷA, first k frames of ŷB]`.

These do not ask adjacent frames to be identical; they ask the reconstruction to reproduce the source transition. This distinction protects genuine cuts and rapid motion.

[*Perceptual Learned Video Compression with Recurrent Conditional GAN*](https://arxiv.org/abs/2109.03082), Ren Yang, Radu Timofte, Luc Van Gool, 2022, IJCAI/preprint, conditions a recurrent discriminator on latents, motion, and recurrent hidden state to encourage temporal coherence. V8 has a 3D PatchGAN but it sees one independently reconstructed GOP at a time; a boundary-specific critic is therefore a **partial analogue**, not duplication.

[*High Visual-Fidelity Learned Video Compression*](https://arxiv.org/abs/2310.04679), Meng Li, Yibo Shi, Jing Wang, Yunqi Huang, 2023 preprint, uses confidence-based feature reconstruction and a periodic compensation loss to address newly exposed regions/checkerboard artifacts. V8 has radio confidence but not flow/reconstruction confidence or periodic compensation. Its objectives target perceptual artifacts, not explicitly reset boundaries, so applicability is **adaptable**.

[*Real-Time Blind Video Temporal Consistency*](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.pdf), Wei-Sheng Lai et al., 2018, ECCV, uses occlusion-masked flow-warping error and perceptual objectives. The source-referenced flow-warped boundary loss is the transferable component; applying blind smoothing after a fade risks freezing or ghosting.

## 6. Content/motion and persistent-scene factorization

[*VidTwin: Video VAE with Decoupled Structure and Dynamics*](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VidTwin_Video_VAE_with_Decoupled_Structure_and_Dynamics_CVPR_2025_paper.html), Yuchi Wang et al., 2025, CVPR, separates structure/global movement from fine details/rapid dynamics. [*Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video*](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bitrate-Controlled_Diffusion_for_Disentangling_Motion_and_Content_in_Video_ICCV_2025_paper.html), Xiao Li et al., 2025, ICCV, learns clip-wise content and frame-wise motion with a low-bitrate bottleneck. These are **speculative/adaptable**, not seam solutions.

For V8, a small persistent appearance vector could encode low-frequency color/illumination/identity while ordinary GOP latents retain motion and fine change. It should not be a free-running generative identity code: at severe compression, such a state can make the decoder preserve an outdated interpretation after a cut. A safer factorization is:

- `8–32` real values for slowly changing global color/appearance statistics;
- optionally a low-resolution spatial feature map derived from the previous reconstruction;
- high-pass/motion content remains entirely current-GOP controlled;
- reliability and scene-change gates decide whether to retain, update, or reset.

Transmitted anchor costs are exact:

| Anchor | V8 budget fraction |
|---:|---:|
| 8 reals/GOP | 0.284% |
| 16 reals/GOP | 0.568% |
| 32 reals/GOP | 1.136% |
| 64 reals/GOP | 2.273% |

The values must replace current latent coordinates unless the modem framing changes. A deterministic RX-only appearance state costs zero RF values; a transmitted anchor synchronizes state but spends scarce payload and remains vulnerable to a selective fade.

## 7. Overlapping-window/context methods

| Method | New-frame stride | RF-rate effect for V8 | Decode memory/latency | Assessment |
|---|---:|---:|---|---|
| One shared source frame, two full V8 codes | 5 | `2816×6/5 = 3379.2` values/s, **+20%** | one reconstructed frame; no required lookahead beyond second code | Directly supported, incompatible with fixed rate |
| Keep 2816/s by truncating each overlap code | 5 | 2346.7 values/window | same | Local prototype lost substantial spatial quality |
| Previous decoded frame as non-transmitted context | 6 | 0 | one frame, no extra lookahead | Best first zero-bit option |
| Previous 2/3 frames as context | 6 | 0 | 0.33/0.5 s history | Better motion estimate; more compute |
| Full previous GOP context | 6 | 0 | six frames | Useful for feature/flow alignment; no RF cost |
| Bidirectional five-GOP latent window | 6 | 0 | one or more GOPs lookahead | No error chain, but added latency/compute and full retraining |

Overlap-add is most useful when the overlapping evidence was produced with different temporal context. Simple RGB cross-fading can reduce a pop while blurring real motion. Feature/latent fusion should be motion-aligned and confidence-weighted, and all metrics must score both transitions adjacent to a shared/interpolated frame because blending can merely move the discontinuity one frame.

## 8. Relevant generative-video techniques

Realistically adaptable:

- StreamingT2V's separation of short transition memory and long appearance memory;
- reuse of previous clean predictions as overlap context;
- shared deterministic anchors/noise analogues to stop ambiguous detail from being regenerated independently;
- motion-aligned feature injection at the first frames of a chunk;
- scene-cut detection and reset.

Mostly impractical:

- diffusion denoising or video-DiT decoding at the receiver;
- large temporal KV caches;
- iterative latent optimization;
- multi-second bidirectional generation;
- stochastic detail regeneration without a TX/RX-shared seed/state.

[*Rethinking Video Tokenization: A Conditioned Diffusion-based Approach*](https://arxiv.org/abs/2503.03708), 2025 preprint, uses a feature cache for arbitrary-length continuity. The cache idea is adaptable, but diffusion decoding is not. [*StreamingT2V*](https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html) is the stronger architectural analogy because it explicitly addresses transitions between chunks.

## 9. Confidence/reliability-aware temporal models

AETV's radio confidence is physically grounded: the production demodulator supplies an MMSE-style `|H|²/(|H|²+N)` weight per received latent coordinate. The decoder already consumes it. Extending this to state is more principled than learning confidence solely from RGB artifacts.

The closest mechanisms are:

- Uni-LVC: scale temporal cues toward intra mode when a reference is unreliable;
- asymmetric-context video JSCC: learn TX/RX conditions from different reconstructed evidence and propagate features independently;
- uncertainty-aware learned compression: model uncertainty in intermediate motion/quantization representations;
- confidence-gated recurrent flow propagation in adjacent video tasks, which suppresses uncertain warped evidence.

A candidate update is not necessarily a scalar convex blend. A practical hierarchy is:

1. hard validity bit (state exists and stream is continuous);
2. scalar trust from pilot SNR and confidence quantiles;
3. low-resolution spatial trust map from the weight grid;
4. learned content novelty/cut gate;
5. feature-wise GRU/adapter update.

Train confidence calibration explicitly: randomly permute confidence, use confidently wrong state, and feed low confidence with clean state. Otherwise the network may ignore the control input or learn SNR shortcuts.

## 10. Metrics for GOP seam error

### 10.1 Boundary-specific core

For each true boundary `b=6,12,…`, define:

`e_t^D = D(ŷ[t+1]-ŷ[t], x[t+1]-x[t])`.

Then report:

- `Boundary Excess Error = mean(e_b) - mean(e_t : t not boundary)`;
- `Boundary Error Ratio = mean(e_b) / mean(e_t : t not boundary)`;
- per-sequence paired confidence intervals, not only a pooled mean;
- continuous-shot and true-cut strata separately;
- low-pass Y/Cb/Cr step error;
- boundary acceleration on both triplets touching the join;
- gradient-domain and perceptual signed-delta error;
- motion-compensated/occlusion-masked boundary error;
- decoder-feature trajectory jump and global reconstruction-bias jump.

The ideal ratio is near 1.0, not necessarily zero error. Report the source's own boundary-to-within motion ratio so difficult natural motion is not confused with codec periodicity.

### 10.2 General temporal metrics

- **Delta-LPIPS / tLPIPS-style measures:** LPIPS or learned features applied to signed frame differences. V8 already uses a VGG delta loss; evaluation should use an independently fixed LPIPS network.
- **Flow warping error:** warp adjacent reconstructed frames with source-derived RAFT flow and mask occlusions. It is sensitive to judder/edge shifts but can reward blur.
- **FloLPIPS:** [Danier et al., 2022](https://arxiv.org/abs/2207.08119) weights LPIPS feature errors with temporal flow distortion; developed for interpolation but useful as a secondary boundary metric.
- **MOVIE / Temporal MOVIE:** [Seshadrinathan and Bovik, 2010](https://live.ece.utexas.edu/research/Quality/movie.html) evaluates distortions along motion trajectories. It is computationally heavier and not boundary-specific.
- **FS-MOVIE:** adds flicker-sensitive spatiotemporal spectral features; useful for perceptual flicker validation.
- **VMAF:** include only as a whole-video secondary metric. Temporal pooling can dilute a one-frame seam, so it cannot be the decision gate.
- **Learned whole-video metrics (FVD/VBench/etc.):** distributional/generative quality measures are too coarse for a six-frame periodic boundary unless also windowed around each seam.

### 10.3 One-hertz signature

At 6 fps, a boundary every six frames is a 1 Hz periodic event. For each scalar trace—mean Y/Cb/Cr, adjacent-frame LPIPS, signed-delta error, low-pass residual energy, decoder feature mean, and reconstruction bias—remove scene trends, window the sequence, and compute a periodogram. Report:

- power at exactly 1 Hz and harmonics 2/3 Hz;
- local noise-floor ratio around 1 Hz;
- phase-locked average by frame index modulo six;
- source-normalized excess spectrum: reconstructed-error spectrum minus source spectrum;
- significance from circularly shifting GOP markers within each video.

With only six samples/s, 3 Hz is Nyquist. Use videos long enough for useful 1 Hz resolution (at least 60 s), do not zero-pad short clips and treat the interpolated curve as new evidence, and stratify true scene cuts. FS-MOVIE's spatiotemporal flicker bands support the general spectral rationale, but the exact 1 Hz boundary-lock statistic is an AETV-specific metric.

## 11. Direct comparison table

`0` under extra transmitted values means the adaptable mechanism can be local state/context; it does not mean the original paper's complete codec has no bitstream. “Main V8” excludes uncommitted worktree prototypes.

| Paper | Year | Technique | Problem solved | Direct GOP work? | AETV analogue | Already in V8? | Modification required | Extra transmitted values | Persistent state | Error-propagation risk | Latency | Compute | Expected seam benefit | Suitability |
|---|---:|---|---|---|---|---|---|---:|---|---|---|---|---|---|
| Perceptual NVC with Video VAE | 2026 | Shared frame, dual reconstruction, averaging | Independent-GOP discontinuity | **Yes** | Overlap decoded transition | GUI blend only, not dual evidence | Stride-5 encoding and fusion | +563.2/s, **+20%** | No | Low | Low | +20% encoder/decoder throughput | Medium-high | Medium; RF rate fails |
| Free-GVC | 2026 | Frame overlap + latent fusion | Inter-GOP flicker/misalignment | **Yes** | Fuse overlapping V8 latents/features | No | New latent fusion and overlapping schedule | V8 1-frame: **+20%** | Short overlap | Low | Low/moderate | Very high full diffusion; small fusion alone | High | Low for full method; medium for fusion idea |
| Rippel et al. | 2019 | Arbitrary learned codec state | General learned prediction/RD | No | Cross-GOP hidden feature | No | Stateful encoder/decoder | 0 adaptable | Yes | High unless reset | Low | Moderate | Medium | Medium |
| NVC joint spatial-temporal | 2020 | ConvLSTM prior | Entropy and motion correlation | No | Compact ConvGRU state | No | Recurrent feature block | 0 adaptable | Yes | Medium-high | Low | Moderate | Medium | Medium |
| LSTVC | 2022 | I-frame initialized, motion-aligned temporal prior | Long-range RD | Within GOP only | Dynamic retain/discard memory | No | Small cross-GOP prior; remove entropy-specific parts | 0 adaptable | Yes | Medium | Low | Moderate/high | Medium | Medium |
| DCVC-DC | 2023 | High-quality long-term contexts, offset diversity | Long-range RD/quality degradation | No | Stable feature reference + alignment | No | Motion/context redesign or small adapter | 0 adaptable | Yes | High | Low | High | Medium-high | Medium as inspiration |
| PLVC recurrent GAN | 2022 | Recurrent conditional discriminator | Perceptual/temporal realism | No | Boundary critic | 3D GAN is partial | Train critic on two-pass joins | 0 | Training state only or recurrent | Low for training-only critic | 0 runtime | Training only | High for texture-phase pops | High as Class I extension |
| HVFVC | 2023 | Confidence reconstruction + periodic loss | Emerged regions/checkerboard artifacts | No | Reliability-aware feature correction | Partial: radio weights only | Boundary periodic loss; feature confidence | 0 | Optional | Low-medium | Low | Moderate | Medium | Medium-high |
| DeepWiVe | 2022 | GOP JSCC, learned residual, bandwidth allocation | Wireless video/cliff effect | No | AETV JSCC baseline | Partial | Fundamental codec/rate allocator | Variable | GOP-local | Low across GOP if reset | Moderate | High | Unknown | Medium context, low direct evidence |
| Asymmetric-context video JSCC | 2026 | Separate TX/RX contexts + feature propagation | State mismatch/error accumulation | No | Train clean-TX and noisy-RX context compatibility | No | Dual context branches or RX-only adapter | Motion side info in paper; 0 for RX-only adaptation | Yes | Explicitly addressed, still present | Low | High full; moderate adapter | High | **High conceptual fit** |
| Robust multipath video JSCC | 2026 | OFDM + conditional warped context + denoiser | Multipath fading robustness | No | AETV OFDM/channel curriculum | Partial | Fundamental codec redesign | Not directly comparable | Reference frames | Medium-high | Moderate | 505.4 GFLOPs reported | Unknown boundary benefit | Low implementation fit |
| Uni-LVC | 2026 | Reliability classifier scales temporal cues toward intra mode | Bad temporal references | No | Confidence gate/bypass | Decoder has confidence, no temporal cue | Gate previous-state adapter | 0 | Reference context | Low if intra fallback works | Low | Low-moderate | High under fades | **High** |
| Uncertainty-aware DVC | 2024 | Ensemble predictive uncertainty | Motion/quantization uncertainty | No | Spatial reliability map | No | Ensemble or cheaper uncertainty head | 0 adaptable | Reference context | Medium | Low | High with ensemble | Medium | Medium |
| StreamingT2V | 2025 | Short/long memory + overlap blending | Hard cuts between generated chunks | Strong analogue | Boundary feature + appearance anchor | No | Small context modules; omit diffusion | 0 for context, overlap otherwise | Yes | High in original AR chain | Low/high depending variant | Very high original | High | Medium as architecture source |
| Consistent video-depth inference | 2025 | Reuse prior predictions as context | Sliding-window flicker | Strong analogue | Previous decoded frames as context | No | Context input, overlap training | 0 | Short | Medium | Low | Moderate/high original | High | High principle, different task |
| VidTwin | 2025 | Structure/dynamics factorization | Compact video representation | No | Appearance/dynamics split | No | New heads/latent allocation | 0 deterministic or 8–64 anchor reals | Optional | Medium | Low | Moderate/high | Medium | Speculative |
| FloLPIPS | 2022 | Flow-weighted perceptual metric | Interpolation temporal quality | Metric only | Boundary evaluation | No | Evaluation implementation | 0 | No | None | Eval only | High eval | N/A | High as secondary metric |

## 12. Proposed AETV experiments

All experiments use paired source sequences and identical channel realizations. Minimum evaluation cells: codec-only clean; production modem clean; AWGN 12/6/0 dB; good multipath (MPG); poor multipath (MPP 12/6/0); measured 40 m path/replay where available; good→deep-fade→good; dropped-GOP→recovery; true cuts; midstream acquisition. Report PSNR/SSIM/LPIPS for guardrails, but promote only on boundary metrics plus clean/channel LPIPS gates.

### Experiment 1 — Two-GOP boundary-aware training (Class I, priority 1)

- **Code:** extend the data spec to 12 contiguous frames; reshape to two six-frame GOPs; call the same released encoder/channel/decoder twice; concatenate only for loss calculation. Add source-referenced boundary delta, low-pass Y/C, gradient-delta, and acceleration terms.
- **Training:** warm-start released V8; 12 frames/example; mixed clean/waveform cells; start with all architecture frozen except decoder, then compare full-model fine-tune.
- **State/reset:** none; exact runtime independence retained.
- **Budget/modem/ONNX:** no change; 2,816 values/GOP; same interfaces.
- **Runtime:** unchanged GPU/CPU. Training roughly 2× forward memory/compute, reducible by batching two GOPs.
- **Expected:** medium seam reduction, minimal average-quality risk.
- **Ablations:** within-only versus boundary; pixel versus perceptual; source-referenced versus naive smoothness; weights 1/2/4/8.
- **Failure:** spends capacity making GOP origins look alike, blurs true motion, or moves error to frame 1.

### Experiment 2 — Random artificial reset boundaries (Class I)

- **Code:** sample 12–24 contiguous frames and split into independent calls at randomized valid positions; pad/mask to six-frame calls or vary crop origin while maintaining exact inference calls.
- **Training:** mix fixed modulo-six, random crop phase, and randomized reset curricula. Always include production six-frame boundaries.
- **State:** none.
- **Interfaces/cost:** unchanged runtime; higher training I/O.
- **Expected:** reduces fixed-position decoder bias and prevents the network learning a special “frame 0 look.”
- **Ablations:** fixed, random, 50/50 mixed; random scene cuts excluded/included with cut labels.
- **Failure:** variable-length emulation differs from actual six-frame geometry; use it only as augmentation.

### Experiment 3 — Boundary-weighted temporal loss (Class I)

- **Code:** implement `L = L_current + λ_b L_boundary` with `λ_b={1,2,4,8}`; include both delta and second-order terms.
- **Training:** same 12-frame independent-pass pipeline as Experiment 1.
- **Cost/interfaces:** no runtime change.
- **Expected:** identifies whether underweighting or missing observation is dominant. Prediction: 2×–4× best; 8× may soften the first new frames.
- **Metrics:** boundary ratio/excess, delta-LPIPS, low-pass step, flow-warped error, within-GOP error, LPIPS/PSNR.
- **Failure:** blur, motion attenuation, over-correction around cuts.

### Experiment 4 — Previous reconstructed frame conditioning (Class II)

- **Code:** add a small residual adapter after the existing decoder. Inputs: current six-frame reconstruction, previous last reconstructed frame, and confidence summaries/maps. Taper the predicted correction to zero after 2–4 frames; exact bypass when invalid.
- **Training conditions:** (a) clean prior, (b) channel-corrupted prior, (c) confidence-gated prior; include deliberately mismatched/corrupted prior as a negative example.
- **State/reset:** one RGB frame plus confidence; reset on any stream discontinuity.
- **Budget/modem:** zero RF values; no modem change.
- **ONNX/runtime:** new optional input/output or separate adapter model. Low GPU/CPU cost if no optical flow.
- **Expected:** high low-frequency seam reduction; modest texture/motion benefit.
- **Risk/recovery:** bounded to the current GOP by the taper; exact recovery after reset.
- **Failure:** ghosting or holding stale appearance after cuts.

### Experiment 5 — Context without duplicate transmission (Class II)

- **Code:** let the new-GOP adapter/decoder consume 1, 2, 3, or 6 prior reconstructed frames while transmitting only the new GOP. Compare concatenation, cross-attention at `1/8` resolution, and motion-aligned feature fusion.
- **Training:** 18-frame/three-GOP sequences; random context resets; clean/noisy/missing context.
- **Budget:** zero RF values.
- **Latency:** no lookahead; history only. Memory 1–6 RGB frames or low-resolution features.
- **Compute:** grows with context; RAFT is likely GPU-suitable but a CPU concern. Test learned lightweight flow or no-flow attention.
- **Expected:** 2–3 frames should outperform one frame for motion; full GOP helps appearance/alignment but may saturate.
- **Failure:** unaligned context softens details; CPU deadline miss.

### Experiment 6 — Compact persistent appearance state (Classes II and III)

- **Code:** pool low-frequency decoder features into `d={8,16,32,64}` values; condition a small affine/FiLM or cross-attention adapter. Separate slow appearance from current motion with high-pass or temporal-residual losses.
- **Deterministic arm:** derive state from decoded video independently; zero RF cost.
- **Transmitted arm:** replace `d` current latent values; costs 0.284/0.568/1.136/2.273% of the V8 budget.
- **Training:** include scene cuts, illumination changes, and fades; state consistency and current-GOP override losses.
- **ONNX:** explicit state input/output for recurrent form; easy tensor interface.
- **Expected:** brightness/color/identity stability; weaker geometric seam improvement.
- **Risk/recovery:** stale scene identity. Gate update, cap state age, and hard-reset on cuts/gaps.

### Experiment 7 — Recurrent cross-GOP memory (Class II)

- **Code:** compare a vector GRU, low-resolution ConvGRU, recurrent residual block, small state-space block, and 1–4-token temporal transformer memory at the decoder bottleneck. Keep state below a predeclared byte/compute budget.
- **Training:** 4–16 GOP unrolls with truncated BPTT, random resets, fades of 1–3 GOPs, and cut transitions.
- **Budget:** zero RF values for RX state; no modem change.
- **ONNX:** explicit state tensors; GRU/ConvGRU generally simpler than dynamic KV caches. Validate CPU provider support.
- **Expected:** potentially highest general continuity.
- **Risk:** highest error propagation. Measure state impulse response, half-life, recovery GOP count, and long-run drift.
- **Failure:** state poisoning, tune-in dependence, scene persistence, CPU memory bandwidth.

### Experiment 8 — Confidence-gated memory (Class II, publication core)

- **Code:** combine current decoded feature, previous memory, full/pooled confidence grid, pilot SNR, and validity flag. Compare scalar gate, spatial gate, channel-wise gate, and GRU update. Include exact intra/bypass branch.
- **Training:** waveform-channel unrolls with deep fades, frequency-selective notches, false-high/false-low confidence augmentation, missing GOPs, and measured paths.
- **State:** retain old stable state during one bad GOP; update from reliable portions; decay/reset if unreliability persists.
- **Budget:** zero RF values.
- **Runtime:** small gate is cheap; spatial gating at `1/8` resolution is feasible on GPU and probably CPU.
- **Expected:** retains seam gains under good reception while limiting fade poisoning.
- **Recovery:** require candidate to return within 5% of independent baseline one GOP after reset and quantify good→fade→good behavior.
- **Failure:** gate ignores confidence, freezes stale state, or treats scene cuts as fades.

### Experiment 9 — Boundary-aware discriminator (Class I)

- **Code:** feed `[last k frames of independently decoded A, first k of B]`, `k=2 or 3`, to a small 3D critic. Real examples are corresponding continuous source windows. Condition the critic on source motion magnitude and/or confidence; use feature matching and confidence-scale only the non-reference adversarial term.
- **Training:** first freeze the critic on a fixed baseline corpus; then alternate updates with conservative GAN weight. Compare against the exact L1/perceptual boundary losses.
- **Budget/runtime:** no inference change if used only for training.
- **Expected:** best for texture phase, face-detail, and hallucinated-detail pops that pixel losses miss.
- **Failure:** plausible but wrong stable texture, GAN instability, or blur avoidance at the cost of fidelity. Gate promotion on LPIPS and source-referenced delta metrics.

### Experiment 10 — Rate-neutral overlap retraining (Class I/II)

- **Code:** jointly train six-frame windows at stride five with approximately 2,347 values/window, or use a longer window so overlap fraction falls. Fuse shared frames/features and supervise both adjacent transitions.
- **Budget:** exactly 2,816 values/source-second only if the per-window code is reduced/repacked; modem framing changes unless windows are scheduled over multiple GOPs.
- **Latency:** one shared-frame buffer; compute +20% calls unless the architecture emits only new frames.
- **Expected:** tests whether overlap's gain can be learned into the smaller per-window code.
- **Failure:** prior local fixed-rate diagnostic suggests substantial spatial-quality loss; treat as lower priority.

### Experiment 11 — Reliability-aware periodic neural I-frame/reset (Class II/III)

- **Code:** make the stateful adapter fall back to released independent decode when confidence/cut/gap logic fires. Optionally reserve 16–32 reals for a state refresh only on a fixed periodic schedule; without a feedback channel, TX cannot know an RX-only fade, so receiver-triggered refresh is unavailable.
- **Budget:** zero for deterministic scheduled reset; 0.568–1.136% if anchor coordinates are always reserved.
- **Expected:** bounds long-horizon drift.
- **Failure:** scheduled reset recreates a visible seam; include reset transitions in the boundary loss.

## 13. Novelty and publication opportunity

The literature establishes each piece separately:

- independently processed GOPs can produce discontinuities, and overlap/fusion can reduce them;
- learned codecs and video generators use persistent frame/feature memory;
- DeepWiVe and newer systems perform video JSCC with graceful channel degradation;
- recent multipath video JSCC combines OFDM and contextual prediction;
- asymmetric-context JSCC directly addresses TX/RX reconstruction mismatch and error accumulation;
- Uni-LVC suppresses unreliable temporal references.

The unresolved combination is an **explicitly boundary-measured analog video JSCC codec whose independent RF GOPs remain reacquirable, while a zero-bit receiver state improves continuity only when per-latent OFDM reliability says the state is trustworthy**. The publication claim should be framed as:

> Reliability-gated, resettable cross-GOP feature memory for independently framed analog neural video transmission under selective fading.

The contribution needs more than a gate formula. A defensible paper would include:

1. a formal boundary-excess metric and 1 Hz periodicity test;
2. TX-clean/RX-noisy state mismatch analysis;
3. hard reset and bounded-error-propagation design;
4. clean/AWGN/Watterson/measured-HF paired results;
5. good→fade→good recovery curves and midstream acquisition;
6. exact RF-rate, latency, GPU, and CPU accounting;
7. comparison with stateless boundary training, overlap, ungated memory, scalar gating, and spatial confidence gating;
8. full video/user evaluation focused on seam noticeability.

No search can prove a universal negative. As of this review, the closest paper is the 2026 asymmetric-context JSCC preprint, followed by Uni-LVC's reliability scaling and the two direct overlap papers. None of those sources combines AETV's exact feature set or evaluates excess error at independent one-second analog GOP boundaries.

## 14. Final ranking

| Rank | Proposed AETV change | Literature basis | Expected seam reduction | RF bitrate cost | GPU cost | CPU cost | Error-propagation risk | Effort | Novelty |
|---:|---|---|---|---:|---|---|---|---|---|
| 1 | Two-GOP independent-pass boundary training + source-referenced losses | Direct GOP-overlap papers; flow consistency literature | Medium | 0 | None runtime | None runtime | None | Low-medium | Medium in analog JSCC context |
| 2 | RX-only boundary-local previous-feature adapter, confidence bypass, tapered correction | Learned state, context-aware windows, Uni-LVC | High | 0 | Low-moderate | Low-moderate | Low and bounded | Medium | High |
| 3 | Reliability-gated ConvGRU/feature memory with random reset/fade curriculum | Asymmetric JSCC, Uni-LVC, recurrent codecs | High | 0 | Moderate | Moderate | Medium if bounded | High | **Very high** |
| 4 | Motion-aligned previous-GOP feature context | DCVC-DC, LSTVC, video restoration | High | 0 | Moderate (high with RAFT) | High with RAFT | Low-medium | Medium-high | Medium-high |
| 5 | Boundary-specific discriminator plus exact boundary losses | PLVC recurrent discriminator | Medium-high for texture pops | 0 | Training only | None runtime | None if stateless | Medium | Medium-high |
| 6 | Compact 8–32-value deterministic appearance state | VidTwin/content-motion factorization | Medium for color/identity | 0 | Low | Low | Medium | Medium | High |
| 7 | 8–32-value transmitted appearance anchor | Factorized representation + neural I-frame idea | Medium | 0.284–1.136% of 2816 | Low | Low | Lower mismatch, fade risk remains | Medium-high | Medium-high |
| 8 | One-frame overlap and dual decode/fusion | Direct ICLR 2026 and Free-GVC | High at shared seam | **+20% symbols/s** | +20% throughput | +20% throughput | Low | Low model / high modem | Low |
| 9 | Five-GOP bidirectional context decoder | Sliding-window/generative context | High | 0 | High | High | Low, no recurrence | Very high | Medium |
| 10 | Conventional P-frame/motion/residual neural codec | DCVC/LSTVC/DeepWiVe | Potentially high | Large/variable side streams | High | High | High | Fundamental redesign | Low for field |

### Lowest-risk V8.1 experiment

**Two-GOP independent-pass boundary-aware fine-tuning.** It fixes the verified supervision hole without changing latent count, modem framing, ONNX interface, runtime state, latency, or failure containment. Use source-referenced delta/low-pass/acceleration losses and compare 1×/2×/4×/8× boundary weights.

### Best zero-bitrate architectural improvement

**RX-only, boundary-local previous reconstructed feature adapter with confidence bypass and tapered correction.** It uses context the receiver already owns, does not require TX/RX state equality, and can be made exactly equivalent to released independent decode after a reset.

### Best overall method

**Combine boundary-aware two-GOP training with a motion-aligned, reliability-gated previous-feature adapter.** The training-only loss tells the system what the seam event is; aligned feature context supplies missing information; confidence and a finite correction horizon contain fade errors.

### Most novel research direction

**Per-latent-reliability-gated recurrent cross-GOP memory under HF fading, with explicit reset/reacquisition guarantees.** Compare scalar, spatial, and feature-wise gates; asymmetric versus RX-only state; and bounded versus unbounded memory. The core result must be a lower boundary error ratio and suppressed 1 Hz signature across clean, AWGN, multipath, and measured paths, without slower post-fade recovery or LPIPS regression.

## Primary-source bibliography

1. Anonymous, [*Perceptual Neural Video Compression with Video Variational Autoencoder at Low Bitrates*](https://openreview.net/forum?id=nfOnCngWtp), ICLR 2026 submission.
2. Ling et al., [*Free-GVC: Towards Training-Free Extreme Generative Video Compression with Temporal Coherence*](https://arxiv.org/abs/2602.09868), 2026 preprint.
3. Chen et al., [*Deep Joint Source-Channel Coding for Wireless Video Transmission with Asymmetric Context*](https://arxiv.org/abs/2601.06170), 2026 preprint.
4. Xiao et al., [*Robust Deep Joint Source-Channel Coding for Video Transmission over Multipath Fading Channel*](https://arxiv.org/abs/2601.01729), 2026 preprint.
5. Zhang, Yang, and Zhu, [*Uni-LVC: A Unified Method for Intra- and Inter-Mode Learned Video Compression*](https://arxiv.org/abs/2603.05756), 2026 preprint.
6. Tang et al., [*Neural Video Compression with Context Modulation*](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.html), CVPR 2025.
7. Henschel et al., [*StreamingT2V*](https://openaccess.thecvf.com/content/CVPR2025/html/Henschel_StreamingT2V_Consistent_Dynamic_and_Extendable_Long_Video_Generation_from_Text_CVPR_2025_paper.html), CVPR 2025.
8. Shao et al., [*Learning Temporally Consistent Video Depth from Video Diffusion Priors*](https://openaccess.thecvf.com/content/CVPR2025/papers/Shao_Learning_Temporally_Consistent_Video_Depth_from_Video_Diffusion_Priors_CVPR_2025_paper.pdf), CVPR 2025.
9. Wang et al., [*VidTwin: Video VAE with Decoupled Structure and Dynamics*](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VidTwin_Video_VAE_with_Decoupled_Structure_and_Dynamics_CVPR_2025_paper.html), CVPR 2025.
10. Li et al., [*Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video*](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bitrate-Controlled_Diffusion_for_Disentangling_Motion_and_Content_in_Video_ICCV_2025_paper.html), ICCV 2025.
11. Ma et al., [*Uncertainty-Aware Deep Video Compression with Ensembles*](https://arxiv.org/abs/2403.19158), 2024 preprint.
12. Li et al., [*High Visual-Fidelity Learned Video Compression*](https://arxiv.org/abs/2310.04679), 2023 preprint.
13. Li, Li, and Lu, [*Neural Video Compression with Diverse Contexts*](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.html), CVPR 2023.
14. Wang and Chen, [*Exploring Long- and Short-Range Temporal Information for Learned Video Compression*](https://arxiv.org/abs/2208.03754), IEEE TMM/preprint, 2022.
15. Tung and Gündüz, [*DeepWiVe*](https://arxiv.org/abs/2111.13034), IEEE JSAC, 2022.
16. Yang, Timofte, and Van Gool, [*Perceptual Learned Video Compression with Recurrent Conditional GAN*](https://arxiv.org/abs/2109.03082), IJCAI/preprint, 2022.
17. Danier et al., [*FloLPIPS*](https://arxiv.org/abs/2207.08119), 2022 preprint.
18. Liu et al., [*Learned Video Compression via Joint Spatial-Temporal Correlation Exploration*](https://arxiv.org/abs/1912.06348), AAAI/preprint, 2020.
19. Rippel et al., [*Learned Video Compression*](https://openaccess.thecvf.com/content_ICCV_2019/html/Rippel_Learned_Video_Compression_ICCV_2019_paper.html), ICCV 2019.
20. Lai et al., [*Real-Time Blind Video Temporal Consistency*](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.pdf), ECCV 2018.
21. Seshadrinathan and Bovik, [*Motion Tuned Spatio-Temporal Quality Assessment of Natural Videos*](https://live.ece.utexas.edu/research/Quality/movie.html), IEEE TIP 2010.
