# FeatLens governance

This document describes who maintains FeatLens, how decisions are made, and how that can change as
the project grows.

## Maintainers

| Maintainer | GitHub | Role |
|---|---|---|
| Turhan Can Kargın | [turhancan97](https://github.com/turhancan97) | Creator and lead maintainer |

FeatLens is currently maintained by one person, on a best-effort basis alongside research work.
The maintainer reviews and merges pull requests, triages issues, cuts releases, and keeps the
documentation, the PyPI package and the live demo up to date.

## How decisions are made

- **Day-to-day changes** (bug fixes, new registry entries, documentation) are decided in the pull
  request or issue where they are proposed.
- **Larger changes** (a new backend adapter, a new visualization method, changes to the public
  API or to the `[B, L, D, h, w]` feature contract) should start as an issue, so the design can
  be discussed before code is written.
- Discussion happens **in public** on GitHub issues and pull requests. Anyone may comment, and
  arguments are weighed on their merits.
- The lead maintainer makes the final decision when there is no consensus, and explains the
  reasoning on the issue or pull request.

## Project scope

FeatLens is for **inspecting the dense, patch-level features of vision models**: extracting them
from any model and layer, and turning them into readable images and data. Proposals that fit this
scope are welcome. Features outside it, such as training models or explaining a single
prediction through its gradients, belong in other tools; the maintainer may decline them even if
they are well implemented.

## Releases and compatibility

- FeatLens follows [Semantic Versioning](https://semver.org/). Before 1.0, minor versions may
  contain breaking changes; these are always listed in the [changelog](CHANGELOG.md).
- Releases are tagged `vX.Y.Z`, published to PyPI by CI, and archived on Zenodo with a DOI. The
  steps are in [CONTRIBUTING.md](CONTRIBUTING.md#releasing-maintainers).
- When a public function or argument is removed or renamed, the change is noted in the changelog,
  and where practical the old name keeps working with a deprecation warning for at least one
  minor release.

## Becoming a maintainer

Contributors who have made several substantial, well-reviewed contributions and want to help run
the project can be invited to become maintainers by the existing maintainers. New maintainers are
listed in the table above. With more than one maintainer, decisions follow the same process, with
consensus among maintainers replacing the lead maintainer's final call where possible.

## If the project becomes inactive

If the lead maintainer can no longer maintain FeatLens, they will say so in the README and either
hand the repository and the PyPI package to an active maintainer or mark the project as
archived. Because FeatLens is MIT-licensed, anyone may fork and continue it at any time.

## Changes to this document

Changes to governance are proposed in a pull request and announced in the changelog.
