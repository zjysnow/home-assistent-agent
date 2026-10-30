# Project Workflow

- After each completed set of project changes, run the relevant validation and
  create a Git commit before reporting completion. Report blockers rather than
  claiming an unverified change is complete.
- Keep [TODO.md](./TODO.md) synchronized with implemented and verified progress.
- Run `python3 -m unittest discover -s tests -v` for application changes.
- Keep credentials, private household data, recordings, and runtime databases
  out of commits. Do not push without an explicit request.
