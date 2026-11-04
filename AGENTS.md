# AGENTS.md

Instructions for AI agents (and humans) working in this repository. It is a research project on quantising Boussinesq solitons; the PI is Gustav Delius. Start by reading `README.md`, which records the conventions, the layout and the current status.

## Ground rules

- **Label every nontrivial claim:**
  - **(a)** proved in the literature;
  - **(b)** conjectured or heuristic in the literature;
  - **(c)** derived in this project;
  - **(d)** a guess.
- **Never invent references.** Cite an arXiv ID or DOI only if you have read the text or verified the entry. Say whether you read the full text or only the abstract. Full texts are kept in `literature/`, which is git-ignored and may be missing in a fresh clone.
- **Verify everything.** Every formula labelled (c) needs a Python (SymPy/mpmath) check, at high precision where feasible. Each script must end by printing `ALL CHECKS PASSED`.
- **Report failures honestly.** List failed checks, including ones that were later fixed, in the write-up (a "failures recorded" list). Never weaken a check to make it pass without saying so.
- **Ask the PI** before making any choice that depends on preference: which equation to study, regularisation schemes, conventions, or the scope of a write-up.
- **Record pending PI decisions in the repository.** When you put a choice to the PI, add it to "Open decisions" in `README.md`: the options, your recommendation and the commit or write-up that raised it. When the PI decides, move it to "Resolved" with the decision and the commit. Never leave a pending decision only in a chat or in an agent's private memory, since other agents on other machines cannot see it.

## Fixed convention

Unless a document states otherwise, use the **good** equation in this form:

- equation: u_tt = u_xx − u_xxxx − 3(u²)_xx;
- potential and tau function: u = φ_x = 2(log τ)_xx;
- Lagrangian: L = ½φ_t² − ½φ_x² + φ_x³ − ½φ_xx²;
- symplectic form: Ω = ∫dπ∧dφ with π = φ_t. ħ refers to this normalisation.

One-soliton data:

- k = sin α and velocity v = ε cos α;
- ℓ = −ε cos 3α, M = 2k, Π = −Mv, P = (2/3)ε k³ cos α.

The **bad** equation, u_tt = u_xx + u_xxxx + 3(u²)_xx, uses k = sinh β, v = ε cosh β and ℓ = ε cosh 3β.

When you use a source with different conventions, write down the explicit dictionary. For example, for Corrigan–Zambon (arXiv:2306.03772): u_CZ = −3φ and L_CZ = 9L.

## Layout and workflow

| Path | Contents |
|---|---|
| `phase0/` | literature map |
| `phase1/` | classical review (`review.tex`) and `scripts/` |
| `phase2/` | quantisation write-ups (`*.tex`/`*.pdf`) and `scripts/` |
| `corrigan-zambon/` | integrable defects: scripts and README |
| `docs/` | standalone interactive page (GitHub Pages) |
| `paper/` | the final paper |
| `scripts_inventory.md` | verification-script inventory, coverage and recorded failures |

- Write-ups are in **LaTeX**; commit both the `.tex` and the compiled `.pdf`. Build with `pdflatex` run twice. Do not commit `*.aux`, `*.log`, `*.out` or `*.toc`.
- Follow the style of the existing write-ups: labelled claims, script names next to each claim, a summary table of claims and checks, failures recorded, and verified references only.
- Prefer new files over edits. Give new scripts a task-specific prefix (e.g. `oneloop_`, `defect_`, `toda_`). Add a short status line to `README.md` when a piece of work is finished.
- Run the checks with `cd phase1/scripts && for f in *.py; do python3 $f; done` (likewise in `phase2/scripts`; in `corrigan-zambon/` the scripts sit at the top level). Some take minutes.

## Git

- Work on `main` unless told otherwise. Pull before you start, since several agents may share the repository.
- Commit only files you created or changed. Commit and push to `main` at the end of each phase or finished piece of work, without waiting to be asked; pull first.
- Never commit `phase0/literature_map.html` or anything in `literature/`.
- Never rewrite history, stash other people's changes, or switch branches in a shared checkout.
