# PRD B: Neural demagnetizing field in JuliaMag (neural-demag, Julia edition)

| | |
|---|---|
| Status | Draft, 2026-11-23. Starts after the closure of the current JuliaMag stage. Phase F7 also needs PRD A milestone M6. |
| Owner | R. L. Novak (UFSC Blumenau) |
| Source discussion | `docs/prd/sources/chatB.md` (claude.ai, July–November 2026). It contains the approved formal project document (Artefact A), the 7-phase execution map (B) and a Python repository skeleton (C). |
| Scientific base | Cai, Li & Wang, *NeuralMAG: Fast and Generalizable Micromagnetic Simulation with Deep Neural Nets*, arXiv:2410.14986 (2024); code at github.com/Caiyq2019/NeuralMAG |
| Integration | both JuliaMag versions: this repository (inference) and the differentiable fork of PRD A (training through the dynamics) |

## 1. Purpose

Replace the most expensive part of a micromagnetic step, the demagnetizing-field
convolution, with a neural network. Study the replacement scientifically, not just
as an engineering trick.

**The four research axes,** unchanged from the approved project:
1. architectures beyond NeuralMAG's U-Net;
2. a detailed characterization of where and why the network's field differs from
   the exact FFT field;
3. the smallest network with good physical quality;
4. application to vortex and skyrmion lattices, a DMI and perpendicular-anisotropy
   regime that NeuralMAG never saw.

**Publications:** two papers. Paper 1 covers axes 1–3; paper 2 covers
skyrmions/DMI. The formal project (Artefact A) runs for 12 months.

## 2. Decision record: what changed from the source plan

The source plan (chatB) was written for Python: PyTorch, `neuraloperator`, mumax3
and `discretisedfield` for the ground truth, Weights & Biases, TensorRT. On
2026-11-23 the author decided that the project is part of JuliaMag, with the
networks developed and trained in Julia, and integrated in both JuliaMag versions.
Consequences:

| topic | source plan | this PRD |
|---|---|---|
| language / DL stack | Python + PyTorch | Julia: Lux.jl, Optimisers.jl, Zygote/Enzyme, MLUtils.jl, CUDA.jl + cuDNN |
| FNO | `neuraloperator` | NeuralOperators.jl if adequate; otherwise a hand-written spectral-convolution layer over CUFFT (a few dozen lines) |
| ground-truth generator | NeuralMAG's PyTorch FDM, then mumax3 OVF | JuliaMag itself: its demag matches mumax3 to 7.7×10⁻⁷ relative (`gpu_validation/revalidation/README.md`) and runs on the same GPU as the training, with no format conversion. mumax3 and NeuralMAG's FDM remain cross-checks. |
| storage | `.pt` tensors, OVF | HDF5 or JLD2 shards with a MANIFEST; OVF only for interchange with mumax3 |
| experiment tracking | Weights & Biases | TensorBoardLogger.jl or Wandb.jl; one JSON metrics file per run in any case |
| figures | matplotlib | matplotlib (JuliaMag project rule), reading the HDF5/JSON outputs |
| speed benchmark | TensorRT on an RTX 3090 (NeuralMAG Table 2) | CUDA.jl inference timings on an RTX 3090, against JuliaMag's own FFT demag on the same card. Direct comparability with NeuralMAG's TensorRT numbers is weaker; state it, and optionally add an ONNX → TensorRT path if Lux export allows it. |
| where the network runs | inside NeuralMAG's PyTorch LLG loop | inside JuliaMag's LLG as an energy/field term, with the rest of the physics (exchange, anisotropy, DMI, STT, temperature) unchanged |

**Kept from the source plan, verbatim in spirit:**
- the four axes and the two papers;
- the baseline-first strategy;
- the unified evaluation protocol;
- the architectures (U-Net baseline, FNO, Attention U-Net, WNO as cuttable,
  compact U-Nets; ViT out of the first round);
- the cloud plan for training;
- the decision triggers;
- the working method (§9).

## 3. Integration design

### 3.1 A separate package plus JuliaMag extensions

- `NeuralDemag.jl`, a new package (repository `neural-demag`), holds:
  - the models (Lux);
  - data generation (calling JuliaMag);
  - training;
  - the evaluation harness;
  - the benchmarks.
  It keeps the deep-learning dependencies out of the JuliaMag core.
- JuliaMag (this repository) gets a package extension `JuliaMagNeuralDemagExt`,
  loaded when NeuralDemag.jl is.
  - It provides a demag term `NeuralDemagField(model, ps, st)` that the `World`
    accepts in place of the FFT `DemagPlan`:
    `World(mesh, mat; demag = NeuralDemagField(...))`.
  - CPU and GPU through the same dispatch as every other term.
- **Guard mode** (traditional version): every K steps, compute the exact FFT demag
  as well. Log the relative error and, optionally, fall back to FFT above a
  threshold. This makes a neural run self-checking, and the guard data feed axis 2.
- The fork of PRD A gets the same term as a `NeuralTerm <: EnergyTerm`, with reverse
  rules, so the network can be trained through the LLG dynamics (§5, F7).

### 3.2 Model input and output

- **Following NeuralMAG.** A 2-layer soft film; input m of both layers (6
  channels); output H_demag (6 channels). Random shape masks and material
  parameters as in the paper.
- **Physical units.** NeuralMAG uses cgs (Oe, emu/cc). JuliaMag uses SI, with B in
  tesla. Convert once, at the data boundary, with tested conversions
  (H[Oe] = 10⁴ B[T] for B = μ0 H). Test the axis order (JuliaMag arrays are
  `(3, Nx, Ny, Nz)`).
- **Normalization.** The symmetric log activation of the target (SLA) and its
  inverse at the output (ISLA), as in the paper's Eq. 18. Port the skeleton's
  `training/losses.py` and its tests.

## 4. Data

**Generator.** `NeuralDemag.gen_dataset` drives JuliaMag on the GPU:
- relax or evolve with the LLG from random states;
- sample (m, B_demag) pairs along the trajectories, as in NeuralMAG Sec. 4.3;
- defaults: 3 nm cells, 2 layers, sizes 32/64/96, 100 simulations × 500 samples per
  size, a random mask fraction of 0.67, in-plane fields of 100–1000 Oe, and
  rotation and mirror augmentation;
- JuliaMag computes B_demag exactly by FFT; it is stored together with m.

**Cross-checks** (once, documented):
1. JuliaMag's B_demag against NeuralMAG's FDM field on identical m (the source plan
   expected a few % at edges, ≤ 5 % inside).
2. JuliaMag against mumax3 on identical m (already 7.7×10⁻⁷; re-check on this
   geometry).

**DMI regime (F5).**
- JuliaMag already has interfacial DMI, PMA and validated skyrmion dynamics.
  Parameters from `configs/data/dmi_skyrmion.yaml` of the skeleton: Ms 580 kA/m,
  A 15 pJ/m, D 2.5–3.5 mJ/m², Ku 0.8 MJ/m³, 2 nm cells, sizes 64/128/256.
- Balance the data: sample transients densely, not only near-relaxed frames.

**MANIFEST per dataset:** code commit, parameters, counts, GPU hours. Data generated
once and reused.

## 5. Phases (adapted from the 7-phase map, Artefact B)

| phase | content (Julia edition) | exit gate |
|---|---|---|
| F0 | **Baseline.** Environment on labmac (CUDA.jl with Pascal sm_61, runtime pinned to 12.x); data generator; Lux port of the NeuralMAG U-Net (Sec. 4.2: 6→6 channels, base width 16, depth 4, leaky ReLU 0.2 in the encoder, dropout 0.2 in the decoder s6/s7, ISLA output); 10 % pilot on the P1000s with measured cost per sample and per epoch; cloud cycle (provision → train → checkpoint → destroy → resume); full dataset; training; reproduction of Table 1 (cooling to InitCore = 5; Square / Random Shape / Random Material) and of the MH criteria (ΔHc ≥ 25 Oe, ΔMr/Ms ≥ 0.03); µMAG #1 | Table 1 within ±2 p.p.; µMAG #1 MH compatible with the paper's Fig. 5; generator cross-checks documented |
| F1 | **Evaluation harness** (`evaluate`): field metrics (MSE, relative-error map, edge vs interior split, radially binned spectral error, magnetostatic-energy error); dynamic metrics (trajectory divergence network vs FFT from the same state, steps to 0.01 / 0.1 divergence, final-state topology with vortex counting by the winding density; physical vortices are merons, ½ each — keep the skeleton's test); observables (Mr, Hc; later skyrmion radius and stability); frozen test set and seeds | harness runs on the baseline in < 1 h; first answers to the three hypotheses (edges, high k, dynamic accumulation) |
| F2 | **Architectures:** FNO (modes 12/16/24; zero-padding test for open boundaries), Attention U-Net, WNO (drop if no green sanity check within 2 sessions), compact U-Nets (base width 4–32 × separable convolutions); 3 seeds each; equal epochs; report parameters, FLOPs and latency; overfit-100-samples sanity check for every new model | master table ≥ 4 families × 3 seeds; Pareto figure; 2–3 winners |
| F3 | **Error anatomy and compression:** where (edges), at what scale (spectra), with what dynamic and energetic consequence; the 10 worst samples per metric; accuracy vs size curve; distillation, structured pruning, fp16 (int8 only if Julia tooling allows; otherwise state it); speed benchmark on a rented RTX 3090 against JuliaMag FFT demag at sizes 32 → 3072 | `docs/analise-erros.md` answers the five questions; speed table; paper 1 figure list frozen |
| F4 | **Paper 1:** "Systematic comparison and error characterization of neural architectures for demagnetizing-field prediction in micromagnetic simulation" | arXiv and submission (npj Comput. Mater. → PR Applied → MLST → JMMM) |
| F5 | **DMI and skyrmions:** DMI dataset; zero-shot test of the soft-film model (error vs ⟨\|mz\|⟩); retraining recipes (DMI only / mixed / fine-tuned); network+LLG with analytic DMI against exact JuliaMag (single-skyrmion radius and profile, lattice stability); speed-up demonstrations (a large skyrmion lattice; vortex gyration and annihilation, linked to the author's permalloy-disc work and to JuliaMag's validated disc) | zero-shot failure quantified; DMI model validated; ≥ 2 demonstrations with measured speed-up |
| F6 | **Paper 2 and release:** "Neural demagnetizing-field prediction for topological magnetic textures: generalization limits and acceleration of skyrmion simulations"; public NeuralDemag.jl with models, datasets with DOIs (the first public DMI dataset of this kind), model cards; JuliaMag extension released | paper 2 submitted; code, data and models public |
| F7 | **Training through the dynamics (with PRD A):** fine-tune the best network with a trajectory loss through the differentiable LLG ("solver in the loop"), and compare field-trained vs dynamics-trained networks on the F1 harness. The source discussion notes that a network can err little in the field and much in the dynamics, or the reverse; little explored in micromagnetics. | a publishable comparison (possible paper 3) |

**Decision triggers** (from the source plan):
- **End of month 2:** if F0 is not closed, cut WNO.
- **End of month 5:** if F2 is not closed, paper 1 goes out without the compression
  section.
- **Month 8:** if paper 1 is not submitted, freeze F5 for 3 weeks.
- **Always:** new ideas go to `docs/ideias-futuras.md`, and the map changes only in
  the weekly review.

## 6. Hardware and cost

- **labmac03/04** (2× Quadro P1000 each, Pascal, 4 GB): development, debugging, the
  10 % pilot, all analysis, data generation of modest size. They are shared
  machines: idle GPUs only, through `run_when_gpu_idle.sh`.
- **Cloud** (vast.ai / RunPod, RTX 3090/4090): full dataset and all training runs.
  - The source estimate is 300–450 GPU-h, about US$ 60–120 for F0–F2, and
    US$ 100–250 for the whole project. Confirm prices at rental time.
  - A Julia + CUDA container image is needed (a provisioning script with Julia, the
    project environment and data sync).
  - Checkpoints are synced off the instance continuously, because spot instances
    die.

## 7. Risks

- **Julia deep-learning ecosystem maturity.** NeuralOperators.jl, quantization and
  TensorRT export are weaker than in Python. Mitigations: hand-written spectral
  layers, fp16 only, CUDA.jl timings.
- **Pascal deprecation.** CUDA 13 dropped sm_61; pin the CUDA runtime to 12.x.
  JuliaMag already runs this way on labmac.
- **NeuralMAG reproduction.** Their protocol details (cooling, masks, units) must be
  matched exactly. Their repository is the reference; run it once in the cloud if
  the Julia reproduction misses Table 1 by more than the tolerance.
- **The 12-month schedule for two papers is tight.** The fallback accepted in the
  formal document: paper 1 submitted, paper 2 in preparation.
- **Shared hardware contention** on labmac (observed November 2026). Plan the
  generation runs in the cloud if the queue is long.

## 8. Deliverables

- `NeuralDemag.jl`: models, data generator, training, evaluation harness, speed
  benchmark, tests.
- `JuliaMagNeuralDemagExt`, with guard mode, in both JuliaMag versions.
- The soft-film and DMI datasets, with MANIFESTs and DOIs.
- Papers 1 and 2 (and possibly paper 3 from F7).
- `docs/`:
  - the project document (Artefact A, kept in Portuguese as approved);
  - the execution map (Artefact B, translated to the Julia stack);
  - the lab notebook;
  - `analise-erros.md`;
  - `protocolo-comparacao.md`.

## 9. Working method (from the source plan; applies to every session)

- **The unit of work is a session of 2–4 h,** with a defined start and end.
- **Every session ends with a commit and a lab-notebook entry**
  (`docs/lab-notebook.md`) whose mandatory field is "próximo passo concreto". The
  next session starts from that field.
- **The 45-minute rule:** stuck for 45 min, stop. Write down what was tried, the
  exact error and the hypothesis, then ask or switch task.
- **Step-by-step guidance:** what to do and how to do it, with concrete commands and
  an objective "done" criterion per step.
- **Plan before execution.** Present the plan for each multi-step task and wait for
  approval. Never invent data; say when something is unknown.
- **Communication** in Brazilian Portuguese; code and repository documents in
  English, except the formal project document.

## 10. How to run this PRD in a new Claude Code session

1. Read this PRD, `docs/prd/sources/chatB.md` (Artefacts A, B and C in full),
   `docs/prd/PRD-A-differentiable-juliamag.md` (the F7 dependency) and JuliaMag's
   `docs/manual.md`.
2. Create the `neural-demag` repository as a Julia package. Port the skeleton's
   tested pieces (losses SLA/ISLA, topology winding count, and their tests). Put a
   copy of the NeuralMAG PDF in `docs/references/`.
3. Present the F0 plan (environment check with `CUDA.versioninfo()` showing the
   P1000 and sm_61, the data generator design, the U-Net port) and ask for approval.
4. Then follow the phase table, with the exit gates as the definition of done.
