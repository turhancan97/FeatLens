# Getting help with FeatLens

FeatLens is maintained on a best-effort basis by a single maintainer (see
[GOVERNANCE.md](GOVERNANCE.md)). All support happens in public on GitHub, so answers help the next
person too.

## Where to look first

- 📖 **Documentation:** <https://turhancan97.github.io/FeatLens/>: installation, usage, every
  visualization method, model sources, and the API reference.
- 🎬 **Video explainer:** a 3-minute walkthrough of how FeatLens works, on the
  [docs home page](https://turhancan97.github.io/FeatLens/#video-explainer).
- 🤗 **Live demo:** <https://huggingface.co/spaces/turhancan97/FeatLens-demo>: try models and
  methods in the browser, no install.
- 🔍 **Existing issues:** someone may have asked already. Search
  [open and closed issues](https://github.com/turhancan97/FeatLens/issues?q=is%3Aissue).

## Asking a question or reporting a bug

[Open an issue](https://github.com/turhancan97/FeatLens/issues/new). Questions are welcome, not
only bug reports. To get a useful answer quickly, include:

- the FeatLens version: `python -c "import featlens; print(featlens.__version__)"`,
- your Python, PyTorch and (if relevant) timm / transformers versions,
- the model spec and the call you made,
- a minimal snippet that reproduces the problem, and
- the full traceback or the unexpected output (a screenshot helps for visual issues).

For a **new model or source** that doesn't load, say where the model comes from (timm,
Hugging Face, `torch.hub`, an external repository, or your own module).

## Feature requests

Open an issue describing what you want to see and why: the research question or workflow behind
it helps decide how it should fit. If you'd like to implement it yourself, say so; the
maintainer will help scope it. See [CONTRIBUTING.md](CONTRIBUTING.md).

## What to expect

- Issues and pull requests are usually answered **within one to two weeks**. Response can be
  slower around holidays and conference deadlines.
- Bugs that break installation or produce wrong feature maps are prioritized over new features.
- Only the **latest release** on PyPI receives fixes. Please check whether a problem persists
  there before reporting.
- There is no private or paid support. Please don't send questions by email or direct message;
  use GitHub issues so the answer stays public.

## Security issues

If you find a security problem (for example, unsafe loading of untrusted model files), please
**don't** open a public issue with the details. Report it privately through GitHub's
[private vulnerability reporting](https://github.com/turhancan97/FeatLens/security/advisories/new)
(the repository's **Security** tab → **Report a vulnerability**). You'll get an acknowledgement
within two weeks, and the fix will be released and credited to you if you wish.

Note that FeatLens loads model weights through timm, Hugging Face and `torch.hub`; only load
checkpoints and hub repositories you trust, since some formats can execute code when loaded.
