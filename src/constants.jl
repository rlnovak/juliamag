# Physical constants (SI units).

"Vacuum permeability [T·m/A]"
const μ0 = 4π * 1e-7

"Bohr magneton [J/T]"
const μB = 9.2740100783e-24

"Boltzmann constant [J/K]"
const kB = 1.380649e-23

"Elementary charge [C]"
const qe = 1.602176634e-19

"""
Gyromagnetic ratio (Landau-Lifshitz), electron [rad/(T·s)]. mumax3 calls this
constant `GAMMA0` in its source, but the value is γLL = 1.7595e11, not μ0·γLL.
The Zhang-Li STT prefactor μB/(2·qe·γLL) uses this value.
"""
const γLL = 1.7595e11

"Reduced Planck constant [J·s]"
const ħ = 1.05457173e-34
