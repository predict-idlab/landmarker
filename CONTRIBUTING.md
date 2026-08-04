# 🤝 Contributing to `landmarker`

🎉 **First off all, thank you for considering contributing to `landmarker`!** 🎉

It's people like you that make `landmarker` useful and successful. There are many ways to contribute, here's a few examples:
* 🐛 [Report bugs](https://github.com/predict-idlab/landmarker/issues/new): Something not working as expected? Please report bugs and we'll try to fix them as soon as possible.
* 🔨 [Fix bugs](https://github.com/predict-idlab/landmarker/issues): We try to fix bugs as soon as possible. If you want to help, please check the [issues](https://github.com/predict-idlab/landmarker/issues).
* 🔍 [Improve documentation](https://github.com/predict-idlab/landmarker/pulls): Did you find a typo in the documentation? Or do you think something is missing? Please help us improve the documentation.
* ✨ [Request/add new features](https://github.com/predict-idlab/landmarker/issues/new): Do you think `landmarker` is missing a feature? Please let us know by creating an issue, however check first if the feature is already requested. Or even better, add the feature yourself and create a pull request.
* 📝 [Write tutorials](https://github.com/predict-idlab/landmarker/pulls): We are always looking for new tutorials, for example how to use `landmarker` for a specific task. Please help us by writing a tutorial and create a pull request.

For more information on contributing to open source projects, [GitHub's own guide](https://opensource.guide/how-to-contribute) is a great starting point if you are new to version control.

## Setting up your development environment

To get started, fork the `landmarker` repository and clone it to your local machine. Then, install the required dependencies using <a href="https://docs.astral.sh/uv/" target="_blank">uv</a> (see <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank">installation instructions</a>):

```bash
uv sync
```

## Running tests
`landmarker` uses [pytest](https://docs.pytest.org/en/stable/) for testing. Coverage reporting is enabled by default:

```bash
uv run pytest
```

To run the complete test matrix across all supported Python versions, as well as the lint and type-check environments, use tox:

```bash
uv run tox
```

You can also run the lint and type checks directly:

```bash
uv run flake8 src/
uv run mypy src/
```

## Building documentation
`landmarker` uses [sphinx](https://www.sphinx-doc.org/en/master/) for documentation, and use MyST markdown for documentation pages. You can build the documentation locally by running the following command:

```bash
uv run sphinx-build docs docs/_build/html
```

We also support the use of [sphinx-autobuild](https://github.com/executablebooks/sphinx-autobuild), which will automatically rebuild the documentation when a change is detected and live-reload the page in your browser. You can run it using the following command:

```bash
uv run sphinx-autobuild docs docs/_build/html --ignore _collections
```

## Ground Rules
The goal is to maintain a diverse community that's pleasant for everyone.
**Please be considerate and respectful of others**.
Everyone must abide by our [Code of Conduct](https://github.com/GenericMappingTools/.github/blob/main/CODE_OF_CONDUCT.md) and we encourage all to read it carefully.
