import os

# This htsne build scales negatively with OpenMP threads on macOS (memory-bound
# attractive-force loop). Single-threaded is ~90x faster here, so cap threads
# before numpy / the compiled extension is imported.
# For your environment, you can try different numbers here or you can comment this out
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

from sklearn import datasets
from hyperbolicTSNE import (
    SequentialOptimizer,
    HyperbolicTSNE,
)
from hyperbolicTSNE.visualization import plot_poincare

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

seed = 42
num_points = 10000
perp = 30

# Match load_data's global-seed behavior for reproducible sampling
np.random.seed(seed)
X_raw, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

# Uniform sample without replacement, sorted indices (matches load_data)
sample_idx = np.sort(
    np.random.choice(np.arange(X_raw.shape[0]), size=num_points, replace=False)
)
X_sampled = X_raw[sample_idx].copy()
y_sampled = y[sample_idx].copy()

# Match load_data: PCA to 100 dims with randomized SVD before D/V
pca_components = min(100, X_sampled.shape[0], X_sampled.shape[1])
X_sampled = (
    PCA(n_components=pca_components, svd_solver="randomized", random_state=seed)
    .fit_transform(X_sampled)
    .astype(np.float32)
)

# Optimizer setting following the original htsne repo
# Just like regular t-SNE, we use early exaggeration with a factor of 12
exaggeration_factor = 12
# We adjust the learning rate to the hyperbolic setting
learning_rate = (X_sampled.shape[0] * 1) / (exaggeration_factor * 1000)
# The embedder is to execute 250 iterations of early exaggeration, ...
ex_iterations = 250
# ... followed by 750 iterations of non-exaggerated gradient descent.
main_iterations = 750

opt_config = dict(
    learning_rate_ex=learning_rate,  # learning rate during exaggeration
    learning_rate_main=learning_rate,  # learning rate main optimization
    exaggeration=exaggeration_factor,
    exaggeration_its=ex_iterations,
    gradientDescent_its=main_iterations,
    vanilla=False,  # if vanilla is set to true, regular gradient descent without any modifications is performed; for  vanilla set to false, the optimization makes use of momentum and gains
    momentum_ex=0.5,  # Set momentum during early exaggeration to 0.5
    momentum=0.8,  # Set momentum during non-exaggerated gradient descent to 0.8
    exact=False,  # To use the quad tree for acceleration (like Barnes-Hut in the Euclidean setting) or to evaluate the gradient exactly
    area_split=False,  # To build or not build the polar quad tree based on equal area splitting or - alternatively - on equal length splitting
    n_iter_check=10,  # Needed for early stopping criterion
    size_tol=0.999,  # Size of the embedding to be used as early stopping criterion
)
opt_params = SequentialOptimizer.sequence_poincare(**opt_config)

htsne = HyperbolicTSNE(
    init="pca",
    n_components=2,
    metric="euclidean",
    hd_method="vdm2008",
    hd_params={"perplexity": perp},
    opt_method=SequentialOptimizer,
    opt_params=opt_params,
    random_state=seed,
    verbose=True,
)
hyperbolicEmbedding = htsne.fit_transform(X_sampled)

fig = plot_poincare(hyperbolicEmbedding, y_sampled)
plt.show()
