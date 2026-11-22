# Early disc comparisons (2026-09-11)

These are the first JuliaMag vs mumax3 comparisons for the single 128 nm × 16 nm disc
with a 40 mA Slonczewski current. They come from before the exchange edge-fill fix
(`bad1f2c`). They were left untracked in `examples/` until 2026-11-22, and were moved
here to keep them.

- `plot_disk_compare2.py`: draws `disk_d1=128nm_t=16nm_40mA_core_traj.png` (core
  trajectory) and `disk_d1=128nm_t=16nm_40mA_m_components.png` (⟨m⟩ components)
  against mumax3.
- `plot_juliamag_new_vs_old.py`: draws `disk_40mA_juliamag_new_vs_old.png`. It shows
  that the table then current was still a pre-fix run: 974.9 MHz in both JuliaMag runs,
  against 909.9 MHz in mumax3.

The scripts read absolute paths under `C:\Users\rlnov\Downloads\resultados projeto ufsc\`,
which are not part of the repository, so they are kept as a record and not meant to be
re-run. The current validation is in `gpu_validation/disk_stt/` (frequencies identical in
both codes since the fix).
