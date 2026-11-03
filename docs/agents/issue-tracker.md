# Issue tracker: GitHub

Issues and specs live in GitHub Issues for
`xiechonghang/LLMarchi_excel`. Use the `gh` CLI from this repo.

## Conventions

- Verify the target before creating issues:
  `gh repo view --json nameWithOwner`.
  The expected target is `xiechonghang/LLMarchi_excel`.
- Create: `gh issue create --title "..." --body-file <file>`.
  Write multiline bodies to a temporary file with real newlines.
- Read:
  `gh issue view <number> --json number,title,body,labels,comments`.
- List:
  `gh issue list --state open --json number,title,body,labels`.
  Apply label and state filters as needed.
- Comment: `gh issue comment <number> --body-file <file>`.
- Apply or remove labels:
  `gh issue edit <number> --add-label "..."` or
  `gh issue edit <number> --remove-label "..."`.
- Close: post the explanation as a comment, then run
  `gh issue close <number>`. Post comments separately even when
  an issue is already closed.
- Sub-issues: use `gh issue create --parent <parent> ...` or
  `gh issue edit <parent> --add-sub-issue <child>`.
  If unavailable, add `Part of #<parent>` to the child body and
  link the child from a task list in the parent.
- Dependencies: use
  `gh issue edit <child> --add-blocked-by <blocker>`.
  If unavailable, add `Blocked by: #<number>` to the child body.
  Work can start when every blocker is closed.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill terminology

- "Publish to the issue tracker" means create a GitHub issue.
- "Fetch the relevant ticket" means read the GitHub issue.

## Wayfinding operations

- Map: one issue labelled `wayfinder:map`, containing
  Notes, Decisions-so-far, and Fog.
- Child tickets: link them as sub-issues, or use the fallback
  parent links above. Use `wayfinder:<type>` labels:
  `research`, `prototype`, `grilling`, or `task`.
- Frontier: take the first open, unassigned child in map order
  with no open blockers.
- Claim: assign the issue with
  `gh issue edit <number> --add-assignee @me`.
- Resolve: comment with the result, close the child, and append
  a summary and link to the map's Decisions-so-far.
