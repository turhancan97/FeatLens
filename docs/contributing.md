# Contributing

**Contributions are welcome — this is an open-source project and we're happy to accept and
support them.** Whether it's a new model adapter, a visualization method, a bug fix, documentation,
or just a question, please jump in. This page mirrors the repository's
[`CONTRIBUTING.md`](https://github.com/turhancan97/FeatLens/blob/main/CONTRIBUTING.md).

## Ways to contribute

- 🐛 **Found a bug or have an idea?** [Open an issue](https://github.com/turhancan97/FeatLens/issues).
  Bug reports, feature requests, and questions are all welcome.
- 🔧 **Want to send a change?** [Fork the repo](https://github.com/turhancan97/FeatLens/fork),
  create a branch, and [open a pull request](https://github.com/turhancan97/FeatLens/pulls).
  Small, focused PRs are easiest to review.
- 💬 **Not sure where to start?** Open an issue describing what you'd like to do and we'll help you
  scope it. Good first contributions include a new entry in the
  [model registry](models.md), a new [visualization method](methods.md), or docs improvements.

## Development setup

```bash
git clone https://github.com/turhancan97/FeatLens.git
cd FeatLens
pip install -e ".[all]"        # editable install with every backend extra
pip install pytest mkdocs-material mkdocstrings[python]
```

## Before you push

- Run the test suite: `pytest -q`.
- For documentation changes, build the site strictly: `mkdocs build --strict`.
- New behavior should come with a test; a new model should be verified to **load and forward**
  to a valid `[B, L, D, h, w]` stack (see `tests/test_smoke.py`).
- Keep changes focused and match the surrounding style.

## License

By contributing you agree that your contributions are licensed under the project's
[MIT License](https://github.com/turhancan97/FeatLens/blob/main/LICENSE).
