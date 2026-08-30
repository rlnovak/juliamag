# JuliaMag — project rules

## Plotting

Always make plots with **Python + matplotlib**, never with Julia (no Plots.jl,
no Makie/CairoMakie for figures the assistant generates). This applies to
comparison plots, result figures, and any chart produced while helping with this
project. Save figures where the task asks; when a task says "the project's
`examples` folder" it means `C:\Users\rlnov\Projetos\mumag\examples`.

(The Julia example drivers in the repo may still contain their own plotting via
`makie_shim.jl` for standalone use, but assistant-generated analysis/comparison
figures use matplotlib.)
