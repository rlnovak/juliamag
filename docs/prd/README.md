# PRDs of the future JuliaMag projects

Product requirement documents for the two projects that follow the closure of the
current stage (see `../roadmap.md`, "Future directions"). Each PRD is written to be
handed to a fresh Claude Code session: it states the goal, the scope, the
architecture, the validation, the milestones and the definition of done, without
relying on this repository's chat history.

| PRD | project | status |
|---|---|---|
| [`PRD-A-differentiable-juliamag.md`](PRD-A-differentiable-juliamag.md) | A. Differentiable JuliaMag (fork with automatic differentiation) | draft 2026-11-23, from `sources/chatA.md` |
| [`PRD-B-neural-project.md`](PRD-B-neural-project.md) | B. Neural demagnetizing field (neural-demag, Julia edition), integrated in both JuliaMag versions | draft 2026-11-23, from `sources/chatB.md` |

The design discussions took place in claude.ai chats, which a Claude Code session
cannot open (shared chats are rendered in the browser). Their exported text is in
`sources/` on the author's machine. It is not committed (personal context), so a
session on another machine needs a copy of that folder.

Dependency: PRD B phase F7 (training through the dynamics) needs PRD A milestone
M6 (`NeuralTerm` and solver-in-the-loop API).
