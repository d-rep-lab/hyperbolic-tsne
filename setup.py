import os
import platform

import numpy
import setuptools
from setuptools import setup, Extension

from Cython.Build import cythonize

# numpy and Cython are guaranteed to be importable here because they are
# declared as build dependencies in pyproject.toml's [build-system] table.
numpy_include = numpy.get_include()


def get_openmp_flags():
    """Return ``(extra_compile_args, extra_link_args)`` enabling OpenMP.

    OpenMP is used by the Cython extensions (``prange``) to run on multiple
    cores. It can be disabled by setting the ``HYPERBOLIC_TSNE_NO_OPENMP``
    environment variable, which is useful on platforms where an OpenMP runtime
    is not available. Without OpenMP the extensions still build and work, they
    just run single-threaded.
    """
    if os.environ.get("HYPERBOLIC_TSNE_NO_OPENMP"):
        return [], []

    system = platform.system()

    if system == "Windows":
        return ["/openmp"], []

    if system == "Darwin":
        # Apple's clang needs a separately-installed libomp (e.g. via
        # `brew install libomp`). Only enable OpenMP if we can find it,
        # otherwise linking would fail.
        for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.isdir(prefix):
                return (
                    ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                    [f"-L{prefix}/lib", "-lomp"],
                )
        return [], []

    # Linux / other unix: gcc and clang support -fopenmp out of the box.
    return ["-fopenmp"], ["-fopenmp"]


openmp_compile_args, openmp_link_args = get_openmp_flags()

extra_compile_args = ["-O3"] + openmp_compile_args
extra_link_args = list(openmp_link_args)

# Suppress the noisy "using deprecated NumPy API" warnings without switching to
# an API version the code does not target.
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

extensions = [
    Extension(
        "hyperbolicTSNE.hyperbolic_barnes_hut.tsne_utils",
        sources=["hyperbolicTSNE/hyperbolic_barnes_hut/tsne_utils.pyx"],
        language="c++",
        include_dirs=[numpy_include],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "hyperbolicTSNE.hyperbolic_barnes_hut.tsne",
        sources=["hyperbolicTSNE/hyperbolic_barnes_hut/tsne.pyx"],
        language="c++",
        include_dirs=[numpy_include],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="hyperbolic-tsne",
    description="Hyperbolic implementation of t-SNE",
    long_description=readme(),
    long_description_content_type="text/markdown",
    version="0.1.0",
    # license="BSD-3-Clause",
    # author="Hunter van Geffen",
    # author_email="huntervangeffen@gmail.com",
    url="https://github.com/chadepl/hyperbolic-tsne",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        # "License :: OSI Approved",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=setuptools.find_packages(include=["hyperbolicTSNE", "hyperbolicTSNE.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scikit-learn>=0.20",
        "scipy",
        "tqdm",
    ],
    extras_require={
        "plot": [
            "pandas",
            "matplotlib",
            "seaborn",
        ],
        "anndata": "anndata",
        "hnsw": "hnswlib>=0.8.0",
        "pynndescent": "pynndescent>=0.5.0",
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
