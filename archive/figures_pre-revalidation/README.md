# Figures from before the 2026-11 re-validation

These are the figures of the validation problems as they stood before the
re-validation of November 2026. They are kept for reference and are no longer
used by the paper or the READMEs. That re-validation re-ran every problem after:

- the exchange edge-fill fix (`bad1f2c`, `0b9110d`);
- the demag-energy fill fix (`9055899`);
- the ⟨m⟩ and vortex-core tracker fixes (`3daa142`, `01a62ff`, `39f2f9e`);
- the rewrite of the GPU path as per-term kernels with Float32 support (`36a619c`).

Each file keeps its original name with the date of the commit that last
produced it appended, under a subfolder that mirrors its original location
(`examples/`, `paper/figures/`, `gpu_validation/std4/`). The tables behind these
figures are in the git history at those commits. The current figures and
tables are at the original paths. The re-validation campaign is
`gpu_validation/revalidation/`.
