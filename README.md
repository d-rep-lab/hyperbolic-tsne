# Accelerating hyperbolic t-SNE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the paper:
> Skrodzki, M., van Geffen, H., Chaves-de-Plaza, N.F., Höllt, T., Eisemann, E. and Hildebrandt, K., "Accelerating hyperbolic t-SNE", 2024, IEEE TCVG ([DOI: 10.1109/TVCG.2024.3364841](https://ieeexplore.ieee.org/document/10432970)).

![teaser of the paper](teaser.png)

If you use our code in your publications please consider citing:
```
@article{skrodzki2024hyperbolic,
    title={Accelerating hyperbolic t-SNE},
    author={Skrodzki, Martin and van Geffen, Hunter and Chaves-de-Plaza, Nicolas F. and H\"{o}llt, Thomas and Eisemann, Elmar and Hildebrandt, Klaus},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    year={2024},
    volume={30},
    number={7},
    pages={4403--4415},    
    doi={10.1109/TVCG.2024.3364841},
    eprint={[TODO](https://ieeexplore.ieee.org/document/10432970)}
}
```

Please find the published version of the paper [here](https://ieeexplore.ieee.org/document/10432970).

## Setup

To execute this setup procedure, we assume that you have a C++ compiler installed on your system that supports at least C++11 and OpenMP.

The package supports Python 3.9 through 3.14. It builds against modern toolchains (numpy 2.x, Cython 3.x); see the [Updates](#updates) section for details on what changed.

1. Create and activate a virtual environment with Python `>=3.9` (we test on 3.14):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # on Windows: .venv\Scripts\activate
   ```
   A conda environment (`conda create --name=htsne python=3.14 && conda activate htsne`) works equally well; use whichever you prefer.
2. Install dependencies with pip: `pip install -r requirements.txt`
3. Install the hyperbolic-tsne package: `pip install .` (this compiles the Cython extensions automatically; the build dependencies in `pyproject.toml` are installed in an isolated build environment). For development, use an editable install instead: `pip install -e .`.
4. To test installation run `python -c "from hyperbolicTSNE import HyperbolicTSNE"`. No errors should be raised and you should see the output `Please note that 'empty_sequence' uses the KL divergence with Barnes-Hut approximation (angle=0.5) by default.`.
5. To re-create the teaser image of this repository, `cd experiments_and_plots` and run `python plot_tree_teaser.py`. The script reads the embedding data and labels from the `teaser_files` folder, plots the teaser image, and saves it to the `teaser_files` folder. It uses paths relative to the `experiments_and_plots` folder, so it must be run from inside that folder.

Note 1:
On macOS, Apple's `clang` does not ship with OpenMP. The build detects a Homebrew `libomp` installation automatically (`brew install libomp`) and enables OpenMP if it is found; otherwise it falls back to a single-threaded build, which is correct but slower during the optimization iterations. You can force-disable OpenMP by setting the `HYPERBOLIC_TSNE_NO_OPENMP` environment variable before installing.

Note 2:
When replicating the teaser image of the repository, depending on your random choice, the image you create might highlight a different point in the left embedding than what is shown in the teaser.
We encourage you to change the seed and render several such images.
The right-hand side will always show the same embedding, but the left-hand side will give you the query structure of the tree for different vertices.
Thereby, you can see which regions are approximated (showing larger cells of the polar quadtree) and which are drilling down to the individual points (showing smaller cells of the polar quadtree).

## Updates

The build was modernized to make the package installable on current Python versions (tested on **Python 3.14**), which required moving off the toolchain the original code targeted:

- **Build system (`setup.py`, new `pyproject.toml`).** Removed the `distutils` imports, which no longer exist in Python 3.12+. Added a `pyproject.toml` declaring the build dependencies (`setuptools`, `wheel`, `Cython>=3.0`, `numpy>=2.0`) so they are available in an isolated build environment. `setup.py` now runs the `.pyx` sources through `Cython.Build.cythonize` (the previous build step that invoked Cython was commented out, so the extensions were never actually compiled) and wires up numpy's C headers and cross-platform OpenMP flags. A separate `python setup.py build_ext --inplace` step is no longer needed — `pip install .` builds everything.
- **Cython 3 compatibility (`.pyx` sources).** Python 3.14 needs Cython 3.x, which is stricter than the Cython 0.29 the code was written for. In `hyperbolicTSNE/hyperbolic_barnes_hut/tsne.pyx`, `2 ** self.n_dimensions` now yields a `double` (Python semantics) and no longer assigns to an integer field, so it was rewritten as `<SIZE_t>(1 << self.n_dimensions)`.
- **numpy 2.x compatibility (`.pyx` sources).** `np.NPY_DEFAULT`, used when wrapping the quadtree as a NumPy struct array, was removed from numpy 2.0. It was replaced with `np.NPY_ARRAY_DEFAULT`, the modern constant declared in numpy's Cython `.pxd` (which also lets Cython inline it as a C constant instead of a runtime attribute lookup).
- **Dependencies (`requirements.txt`, `extras_require`).** The old exact pins (`numpy==1.23.5`, `scipy==1.10.0`, `Cython==0.29.33`, `hnswlib==0.7.0`, ...) have no wheels for Python 3.14 and could not be built. They were relaxed to minimum-version bounds that provide 3.14 wheels.
- **Teaser script usage.** `experiments_and_plots/plot_tree_teaser.py` uses paths relative to its own folder, so it is now documented to be run from inside `experiments_and_plots`.
- **Removed the Dockerfile.** It targeted the old toolchain (pinned Python 3.9.16 and the deprecated `setup.py build_ext` step) and no longer matched the relaxed dependency bounds. The `venv`/conda + pip instructions above are the supported setup path.

## Data

You can run hyperbolic TSNE on your high-dimensional data. 
Nevertheless, the examples and experiments in this repository rely on specific datasets. 
Below, we provide download links for each. 
We recommend putting all datasets in a `datasets` directory at the root of this repository.
The `load_data` function expects this path (`data_home`) to resolve the dataset.

Individual instructions per dataset:
- [LUKK](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-62)
- [MYELOID8000](https://github.com/scverse/scanpy_usage/tree/master/170430_krumsiek11)
- [PLANARIA](https://shiny.mdc-berlin.de/psca/)
- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [WORDNET](https://github.com/facebookresearch/poincare-embeddings)
- [C_ELEGANS](https://github.com/Munfred/wormcells-data/releases)

## First steps

There are two ways of getting started with the `hyperbolicTSNE` package.
First, `example_basic_usage.ipynb` offers a step-by-step guide showing how to use the HyperbolicTSNE package to embed a high-dimensional dataset. 
Second, the `example_different_params.py` script shows how to set up a script for quick experimentation. In this case, to compare the effect of different parameters.

## Replicating the paper results

This folder contains three types of files:
- Scripts to generate experimental data via embedding different data sets into hyperbolic space. These are pre-fixed with "data generation". 
- Scripts to create plots from the data, as they appear in the publication.
- Scripts to create tables from the data, as they appear in the publication.

The general workflow to reproduce the results from the paper is:
- Run the scripts to generate data.
- Run the scripts to plot the data.
- Run the scripts to generate tables.

Note that the data generation scripts assume a top-level folder, i.e., a folder next to "examples", "experiments", etc., called "datasets" that holds the datasets to be embedded.

## License and third-party software
The source code in this repository is released under the MIT License. However, all used third-party software libraries are governed by their respective licenses. Without the following libraries, this project would have been considerably harder: 
[scipy](https://scipy.org),
[numpy](https://numpy.org),
[scikit-learn](https://scikit-learn.org/stable/),
[hnswlib](https://github.com/nmslib/hnswlib),
[pandas](https://pandas.pydata.org),
[anndata](https://anndata.readthedocs.io/en/latest/),
[seaborn](https://seaborn.pydata.org),
[setuptools](https://github.com/pypa/setuptools),
[Cython](https://cython.org),
[tqdm](https://github.com/tqdm/tqdm),
[ipykernel](https://ipython.org).
