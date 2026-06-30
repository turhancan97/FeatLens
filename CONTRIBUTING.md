# Contributing to FeatLens

**Contributions are welcome — this is an open-source project and we're happy to accept and support
them.** Whether it's a new model adapter, a visualization method, a bug fix, documentation, or just
a question, please jump in. Thank you for helping make FeatLens better! 🎉

## Ways to contribute

- 🐛 **Found a bug or have an idea?** [Open an issue](https://github.com/turhancan97/FeatLens/issues).
  Bug reports, feature requests, and questions are all welcome — include the model spec, a minimal
  snippet, and the full traceback where relevant.
- 🔧 **Want to send a change?** [Fork the repo](https://github.com/turhancan97/FeatLens/fork),
  create a branch, and [open a pull request](https://github.com/turhancan97/FeatLens/pulls).
  Small, focused PRs are easiest to review.
- 💬 **Not sure where to start?** Open an issue describing what you'd like to do and we'll help you
  scope it. Good first contributions include a new entry in the model registry
  (`featlens/registry.py`), a new visualization method (`featlens/methods.py`), or docs improvements.

## Development setup

```bash
git clone https://github.com/turhancan97/FeatLens.git
cd FeatLens
pip install -e ".[all]"        # editable install with every backend extra
pip install pytest mkdocs-material "mkdocstrings[python]"
```

Install PyTorch for your platform first (https://pytorch.org).

## Making a change

1. Create a branch off `main`: `git checkout -b my-change`.
2. Make your change, keeping it focused and matching the surrounding style.
3. Add or update tests for new behavior.
4. Run the checks below.
5. Commit, push your branch, and open a pull request describing the change and why.

## Before you push

- **Tests:** `pytest -q` (the suite uses small models with `pretrained=False`, so it runs on CPU
  without network).
- **Docs:** for documentation changes, build the site strictly — `mkdocs build --strict`.
- **New behavior** should come with a test.
- **A new model** should be verified to **load and forward** to a valid `[B, L, D, h, w]` stack —
  see `tests/test_smoke.py`, whose `test_depth_for_every_registry_model` already iterates the
  registry.

## Reporting a bug

Please include:

- the FeatLens version (`python -c "import featlens; print(featlens.__version__)"`),
- the model spec and the call you made,
- a minimal reproducible snippet, and
- the full traceback.

## License

By contributing you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
