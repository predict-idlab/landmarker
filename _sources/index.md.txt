```{toctree}
:hidden:
:titlesonly:

user/index
examples/index
reference/index
```



```{toctree}
:caption: Development
:hidden:
:titlesonly:
:maxdepth: 1


dev/contributing
dev/changelog
dev/code_of_conduct
dev/license
```

```{toctree}
:caption: API reference
:hidden:

reference/data
reference/datasets
reference/heatmap
reference/losses
reference/metrics
reference/models
reference/schedulers
reference/train
reference/transforms
reference/utils
reference/visualize
```

(main-page)=
<p align="center">
    <a href="https://predict-idlab.github.io/landmarker">
        <img alt="landmarker" src="_static/images/logo.svg" width="66%">
    </a>
</p>

# Landmarker

:::::{caution}
🚧 Still under construction. 🚧
:::::

**Version**: {{ version }}

**Useful links**:
[Instalation instructions](installation) |
[Source repository](https://github.com/predict-idlab/landmarker) |
[Issue tracker](https://github.com/predict-idlab/landmarker/issues) |


Landmarker is a [PyTorch](https://pytorch.org/)-based toolkit for (anatomical) landmark detection in
images. It is designed to be easy to use and to provide a flexible framework for state-of-the-art
landmark detection algorithms for small and large datasets. Landmarker was developed for landmark
detection in medical images. However, it can be used for any type of landmark detection problem.


:::::{grid} 2

::::{grid-item-card} 🚀 Getting started
:link: user/index
:link-type: doc
:text-align: center

New to Landmarker? Start here to learn how to install Landmarker and how to use it.

::::

::::{grid-item-card} 🔦 Examples
:link: examples/index
:link-type: doc
:text-align: center

The examples section contains a collection of examples that demonstrate how to use Landmarker

::::

::::{grid-item-card} 📖 API reference
:link-type: doc
:link: reference/index
:text-align: center

The API reference contains a detailed description of all Landmarker's methods, modules, and
classes.

::::

::::{grid-item-card} 🚧 Development
:link: dev/contributing
:link-type: doc
:text-align: center

The development section contains information for developers who want to contribute to Landmarker.

::::
:::::