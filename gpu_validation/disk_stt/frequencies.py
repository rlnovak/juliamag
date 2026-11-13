"""Reproduce the gyration-frequency table of the 128 nm disk validation from the
versioned tables in ./tables (window 1-7 ns; free gyration 0.5-7 ns).

Usage: python frequencies.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gyrofreq import load, freq_phase, freq_fft  # noqa: E402

T = os.path.join(HERE, "tables")
# (label, file, core-x column, core-y column, t1)  — frequencies are taken from <m>
RUNS = [
    ("Slonczewski + Zhang-Li  mumax3  (reference)", "mumax_ref_es0.txt", 1, 2, 1e-9),
    ("Slonczewski + Zhang-Li  JuliaMag",            "sttzl_es0_dt5.txt", 1, 2, 1e-9),
    ("Slonczewski only        mumax3",              "mumax_stt_es0_noZL.txt", 1, 2, 1e-9),
    ("Slonczewski only        JuliaMag",            "jm_stt_es0_dt5.txt", 1, 2, 1e-9),
    ("free gyration           mumax3",              "mumax_free_es0.txt", 1, 2, 0.5e-9),
    ("free gyration           JuliaMag",            "free_es0_dt5.txt", 1, 2, 0.5e-9),
    ("[es4] Slon+ZL           mumax3",              "mumax_stt_es4.txt", 1, 2, 1e-9),
    ("[es4] Slon only, dt/2   JuliaMag",            "stt_es4_dt25.txt", 1, 2, 1e-9),
]


def decay_rate(fn):
    a = np.loadtxt(fn, comments="#")
    t = a[:, 0]; r = np.hypot(a[:, 1], a[:, 2]); w = t > 0.5e-9
    return -np.polyfit(t[w], np.log(r[w]), 1)[0]


print(f"{'run':45s} {'f_x':>8s} {'f_y':>8s} {'f_FFT':>8s}  [MHz]")
for label, fn, cx, cy, t1 in RUNS:
    t, x, y = load(os.path.join(T, fn), cx, cy)
    fx, fy = freq_phase(t, x, t1, 7e-9), freq_phase(t, y, t1, 7e-9)
    print(f"{label:45s} {fx/1e6:8.2f} {fy/1e6:8.2f} {freq_fft(t, x, t1, 7e-9)/1e6:8.1f}")
for label, fn in [("free gyration mumax3", "mumax_free_es0.txt"),
                  ("free gyration JuliaMag", "free_es0_dt5.txt")]:
    print(f"decay rate {label:24s} {decay_rate(os.path.join(T, fn))*1e-9:.4f} 1/ns")
