# Domain docs

This repo uses a single-context layout.

## Before exploring

- Read `GLOSSARY.md` at the repo root.
- Read ADRs under `docs/adr/` relevant to the area being explored.

If these files do not exist, proceed silently.
The domain-modeling skill creates them lazily when terms or
decisions are resolved.

## Layout

- `GLOSSARY.md`: shared domain terminology.
- `docs/adr/NNNN-short-title.md`: architecture decision records.

## Vocabulary

Use glossary terms when naming domain concepts in issues,
proposals, hypotheses, tests, and implementation.

When a needed concept is missing, reconsider whether it belongs
to the domain. Record genuine vocabulary gaps for domain-modeling.

## ADR conflicts

If a proposal contradicts an existing ADR, identify the ADR and
explain why its decision should be reconsidered.
