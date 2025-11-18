---
applyTo: '**'
---

# Commit Guideline for SecureScribeAdmin

Purpose:
Provide a concise, consistent commit message and branch workflow guideline for contributors and automated agents (including AI) to produce clear history and safe changes.

Principles:
- Keep commits small and focused.
- Use imperative, present-tense subject lines.
- Do not include secrets, encryption keys or sensitive data in commits.
- Exclude local task files (.tasks/*) from commits.

Commit message format:
<type>(scope?): subject

Optional body separated by a blank line. If needed, include:
- Motivation / Summary of change
- Implementation notes
- Migration steps or required follow-ups

Footer for metadata (issue IDs, breaking changes, co-authored-by).

Allowed types (use lower-case):
- feat: new feature
- fix: bug fix
- refactor: code change that neither fixes bug nor adds feature
- docs: documentation only changes
- style: formatting, lint, no code logic change
- perf: performance improvements
- test: adding or fixing tests
- chore: maintenance tasks (deps, tooling)
- build: CI/build system changes
- revert: reverts a previous commit

Examples:
- feat(auth): add JWT refresh token endpoint
- fix(api): handle 500 when token is missing
- refactor(docker): simplify container restart logic
- docs: update README for volumes
- chore: bump python dependencies

Subject rules:
- Max ~72 characters for subject line
- Use imperative mood: "Add", "Fix", "Update"
- No trailing period
- Keep scope optional (module or file area)

Body rules:
- Wrap at ~100 characters
- Explain why, not just what
- Mention migration steps or config impacts
- Never paste secrets or private keys

Branch naming:
- Feature / task branches: task/[TASK_IDENTIFIER]_[YYYY-MM-DD]_[N]
    - TASK_IDENTIFIER: short slug (e.g. admin-backend-docker-management)
    - Example: task/container-config-manager_2025-10-16_1
- Hotfix: hotfix/[short-desc]_[YYYYMMDD]
- Release: release/vX.Y.Z
- Use lowercase, hyphens or underscores as project convention requires

Pull request checklist:
- Link to relevant issue or task file
- Include testing steps and screenshots (if UI)
- Ensure linting (ruff) passes and formatting consistent
- Confirm no secrets checked in
- Ensure volume/network config changes documented
- Update CHANGELOG if feature or breaking change

Committing rules:
- Stage only related changes per commit
- Rebase and squash locally when appropriate for a clean history before merging
- Use signed commits if project policy requires
- Do not commit local .tasks/* files; use: git add --all :!.tasks/*

Automation & CI:
- CI will run lint, tests, and container checks on PRs
- Ensure commits that affect container/network configs include validation steps

Enforcement:
- Maintainers may require rework of commit messages or branch names
- Use conventional commits when possible to enable automated changelogs

Short message guideline:
When a very short commit is appropriate, use the same format but keep subject minimal, e.g.:
- chore: short message

This file should be updated when workflow or CI requirements change.