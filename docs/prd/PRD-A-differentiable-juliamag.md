# PRD A: Differentiable JuliaMag (fork)

| | |
|---|---|
| Status | Draft, 2026-11-23. Starts after the closure of the current JuliaMag stage (`docs/roadmap.md`). |
| Owner | R. L. Novak (UFSC Blumenau) |
| Source discussion | `docs/prd/sources/chatA.md` (claude.ai, 2026-11-23) |
| Upstream | this repository (`rlnovak/juliamag`), the validated reference code |
| Related | PRD B (neural demagnetizing field), which consumes milestone M6 of this PRD |

## 1. Purpose

JuliaMag has two goals, set by the author:
1. a production micromagnetic code, competitive with mumax3 and MicroMagnetic.jl;
2. a differentiable research vehicle: inverse problems, and training neural networks
   through the dynamics.

The current repository delivers goal 1. This project delivers goal 2.

**What it adds.** A version of JuliaMag in which every simulation output (a
trajectory, an equilibrium, a spectrum, a switching field) can be differentiated
with respect to:
- the material parameters: A, D, Ku, Msat(x), α;
- the applied fields and currents as functions of time;
- the geometry;
- the initial state.

**Why Julia.** magnum.np, NeuralMag and SpinTorch already do this in Python
(PyTorch or JAX). The selling point here is not "micromagnetics in Julia" but:
- a differentiable code whose forward path is mumax3-validated, bit for bit against
  the reference JuliaMag;
- composability with SciML: `SciMLSensitivity.jl` adjoints, and Lux.jl networks
  as energy terms.

## 2. Decision record

- **2026-11-23, the author decided on a fork.** The differentiable version is
  developed as a fork of this repository; the traditional code stays the
  reference.
  - The source discussion (chatA, answer 2) argued the opposite: "the production
    code must not be the slow version of the differentiable one, and the
    differentiable version must not be a fork". Its argument: the hand-written
    field kernels are their own differentiation rules, so one code can serve both
    paths.
  - The fork is kept compatible with that argument (§5.1), so that a later merge
    stays possible and cheap:
    - the forward kernels are not rewritten;
    - the new abstractions wrap them;
    - the fork regularly merges upstream.
  - Re-evaluate at milestone M3. If the rules turn out to be thin wrappers that
    cost nothing in the forward path, propose upstreaming them.
- **Zygote is not the AD engine for the kernels.** Micromagnetic code mutates
  pre-allocated buffers and does in-place FFTs, which Zygote handles badly. The
  engine is Enzyme.jl (LLVM level, handles mutation, CUDA.jl support), plus
  hand-written ChainRulesCore rules where Enzyme is immature (FFT).
- **Reverse mode only.** E is a scalar and m has 3N components, with N ~ 10⁵–10⁷.
  ForwardDiff is used only for tiny test systems.

## 3. Key technical insight

Every standard term gives a field that is affine in m:
- exchange: H = −L_ex m;
- demag: H = −N̂ m;
- uniaxial anisotropy: ∝ (m·u)u;
- DMI: a discrete linear operator;
- Zeeman: a constant.

So H_eff = −L m + h_ext with L symmetric, and the Hessian of the energy is L itself.

Consequences:
- **Free VJPs.** The vector–Jacobian product of the field, applied to a cotangent
  λ, is the same kernel applied to λ. The fast hand-written kernel is its own
  reverse rule, at no extra engineering or runtime cost.
- **Parameter derivatives.** The field is linear in A, D and Ku, so ∂H/∂p is the
  same kernel with unit parameter. Msat(x) needs more work; it is the route to
  topology optimization.
- **Demag pitfall.** E_d = (μ0 Ms²/2) mᵀ N̂ m, so ∂E/∂m = μ0 Ms² N̂ m: the ½
  cancels. Naive AD through the FFT pays two convolutions where the direct code
  pays one. A custom rule that reuses the H_d already computed (N̂ symmetric)
  removes this factor-2 cost on the dominant term.
- **The |m| = 1 constraint.** The effective field is the unconstrained gradient;
  the LLG cross products project it. Parametrizing in (θ, φ), or normalizing a
  free vector, makes AD return the projected gradient, which is different. The
  choice must be deliberate and documented.
- **Non-linear terms** (cubic anisotropy, higher-order or magnetoelastic terms) do
  not have this property. They get a hand-written Hessian, or generic AD.

## 4. Scope

**In scope:**
1. **A common term interface,** with the existing kernels behind it:

   ```julia
   abstract type EnergyTerm end
   energy(t::EnergyTerm, m, p)        # scalar: validation, minimizers, NEB
   field!(H, t::EnergyTerm, m, p)     # fast mutating kernel (CPU / CUDA)
   # + reverse rules (ChainRulesCore rrule and/or Enzyme rules) reusing field!
   ```

2. **AD as a test oracle.** For every term, field! must equal −∂E/∂m from
   Enzyme, ForwardDiff or FiniteDifferences on small systems, as automated tests.
   This catches sign, factor-2 and boundary-term errors. These tests are useful
   to upstream too, and should be offered there.
3. **Prototyping path.** A new term can be written as `energy` only, with its field
   from AD; it gets a hand-written kernel once it proves useful.
4. **Differentiable time integration:**
   - the RHS of the LLG (with STT) inside `SciMLSensitivity.jl` adjoints, with
     checkpointing;
   - and/or a discrete adjoint of JuliaMag's own mumax3-style steppers, with
     checkpointed reverse sweeps.
   - Decide by prototype (§7, unknown U2).
5. **Differentiable equilibria.** Gradients of a relaxed state through the implicit
   function theorem (the Hessian L is already available as an operator), instead of
   backpropagating through the minimizer iterations.
6. **Differentiable parameters:**
   - A, D, Ku, Msat (scalar, then per cell), α;
   - B_ext(t) and J(t) as parametrized functions;
   - the initial state.
7. **A neural energy term,** `NeuralTerm <: EnergyTerm`, a Lux.jl model, with
   training through the dynamics ("solver in the loop"). This is the hook for
   PRD B.
8. **Inverse-problem demonstrations and a paper** (§6, M5–M7).

**Out of scope for this PRD:**
- multi-backend GPU (KernelAbstractions is deferred upstream);
- atomistic models;
- the neural-network research itself (PRD B).

## 5. Architecture

### 5.1 Fork hygiene

- Fork from the commit that closes the JuliaMag stage; record it in `UPSTREAM.md`.
- The forward path stays bit-identical to upstream: the whole upstream validation
  suite (`test/`, `gpu_validation/` comparisons) runs in CI on the fork, with
  identical tables required.
- Merge upstream at least at every upstream release. Keep the fork's changes in new
  files where possible: `src/ad/`, `src/terms/`, `src/sensitivity/`.
- The package name is distinct from upstream so both can be installed together
  (e.g. `JuliaMagAD`, to be decided).

### 5.2 Layers

| layer | content | AD mechanism |
|---|---|---|
| terms | `Exchange`, `Demag`, `UniaxialAnisotropy`, `InterfacialDMI`, `BulkDMI`, `Zeeman`, `SlonczewskiSTT`, `ZhangLiSTT`, later `NeuralTerm` | rrule: VJP = kernel on the cotangent (linear terms); demag rule reusing H_d; Enzyme for the rest |
| world | a tuple of terms plus parameters (a `ComponentArray`, so they are differentiable) | — |
| dynamics | LLG RHS `f(m, p, t)` | Enzyme or rules for the VJP; adjoint in SciMLSensitivity, or a discrete adjoint |
| equilibria | the minimizer, plus an implicit-function rule | linear solve with the Hessian operator (CG / MINRES) |
| objectives | losses on m(t), ⟨m⟩(t), spectra, energies, trackers | Zygote or Enzyme at the top level (no mutation there) |

### 5.3 GPU

- CUDA.jl only (Pascal sm_61 on labmac03/04, with the CUDA runtime pinned to 12.x;
  CUDA 13 dropped Pascal).
- Enzyme with CUDA.jl kernels and CUFFT is a known risk (§7, U1). The demag rule is
  written by hand regardless; this is trivial because N̂ is symmetric.

## 6. Milestones and acceptance criteria

| M | deliverable | done when |
|---|---|---|
| M0 | Fork, CI, `UPSTREAM.md`; upstream suite green | all upstream tests and validation tables identical |
| M1 | AD oracle tests for every term | field! = −∂E/∂m to 1e-12 (Float64) on open and periodic meshes, regions, both DMI forms, edge-smoothed geometry; a failing variant (wrong sign, missing boundary term) is detected |
| M2 | The `EnergyTerm` layer over the existing kernels | forward results bit-identical to upstream; no allocation added in the production path; RHS cost unchanged (1.11 ms Float32 on the P1000 disc) |
| M3 | Reverse rules for all terms (CPU and GPU) | VJP checks against finite differences to 1e-6 relative; the demag rule costs one convolution; GPU = CPU |
| M4 | Differentiable dynamics | gradient of a loss on m(T) w.r.t. A, Ms, α, B_ext(t) matches finite differences; memory and time vs N measured up to N = 10⁶ for the adjoint options; choice documented |
| M5 | Inverse demonstrations, synthetic data from the reference code | see below |
| M6 | `NeuralTerm` and solver-in-the-loop training API | a small Lux model trained through 1 ns of dynamics on a toy problem reduces the trajectory loss; API documented for PRD B |
| M7 | Paper on differentiable micromagnetics in Julia | comparison with magnum.np / NeuralMag on a published inverse problem; arXiv preprint |

**M5 demonstrations:**
1. Recover A and Ms from a synthetic FMR or ringdown spectrum.
2. Recover D from skyrmion radius data.
3. Find an optimal current pulse J(t) for vortex-core switching under a fixed
   energy budget.
4. Topology optimization of Msat(x) for a magnonic element, as in SpinTorch.
5. Recovery within the noise level, and gradients cross-checked by finite
   differences.

## 7. Unknowns to prototype first (before committing to designs)

- **U1. Enzyme on CUDA.jl kernels, especially with CUFFT.** A one-week prototype:
  differentiate the exchange kernel and the demag convolution on the GPU. If
  Enzyme fails, hand-written rules everywhere; the linear structure makes this
  cheap.
- **U2. Which adjoint scales for N ~ 10⁶ with an adaptive integrator.** Compare:
  - `InterpolatingAdjoint` with checkpointing;
  - `QuadratureAdjoint`;
  - `BacksolveAdjoint` (unstable for dissipative LLG?);
  - a discrete adjoint of RK45 with checkpoints.
  Measure memory, time and gradient accuracy.
- **U3. The |m| = 1 parametrization** for the objectives that need constrained
  gradients (minimizers, NEB); a test on Stoner–Wohlfarth.
- **U4. Float32 gradients.** The accuracy of adjoints over long runs in single
  precision.

## 8. Risks

- Enzyme and CUDA maturity (U1), with hand-written rules as the mitigation.
- Adjoint memory for long runs (U2): checkpointing, and shorter training windows.
- Divergence from upstream: CI on the upstream suite, and frequent merges.
- Scope creep into PRD B: the neural research stays in PRD B; this PRD delivers
  only the hook (M6).

## 9. References (from the source discussion)

- MicroMagnetic.jl: arXiv:2406.16064; Chinese Physics B (2024),
  doi:10.1088/1674-1056/ad766f.
- magnum.np: Sci. Rep. (2023), arXiv:2302.08843.
- NeuralMag: npj Comput. Mater. (2025), arXiv:2411.11725. Not to be confused with
  NeuralMAG, arXiv:2410.14986 (PRD B).
- SpinTorch (backpropagation through time for spin-wave neural networks).
- Perspectives on inverse design for AI magnonics, arXiv:2607.07324.
- SciMLSensitivity.jl, Enzyme.jl, ChainRulesCore.jl, Lux.jl.

## 10. How to run this PRD in a new Claude Code session

1. Read `docs/roadmap.md`, this PRD, `docs/prd/sources/chatA.md` and the upstream
   paper `paper/juliamag.tex` (method and validation).
2. Present a plan for M0–M1 and ask for approval before executing; the author's
   rule is plan first.
3. Work milestone by milestone. Each milestone ends with:
   - tests;
   - a validation note in `docs/`;
   - a commit;
   - an Obsidian flow note (global CLAUDE.md rule).
4. Physics-rigor rules of the upstream project apply: like-for-like validation,
   subagent review before long GPU runs, idle-GPU-only use of labmac03/04
   (`gpu_validation/run_when_gpu_idle.sh`).
