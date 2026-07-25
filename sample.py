import matplotlib.pyplot as plt

from sklearn import datasets
from hyperbolicTSNE import (
    SequentialOptimizer,
    HyperbolicTSNE,
)
from hyperbolicTSNE.visualization import plot_poincare

X, sr_color = datasets.make_swiss_roll(n_samples=1500, random_state=0)

# Optimizer setting following the original htsne repo
# Just like regular t-SNE, we use early exaggeration with a factor of 12
exaggeration_factor = 12
# We adjust the learning rate to the hyperbolic setting
learning_rate = (X.shape[0] * 1) / (exaggeration_factor * 1000)
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

htsne = HyperbolicTSNE(opt_params=opt_params)
hyperbolicEmbedding = htsne.fit_transform(X)

fig = plot_poincare(hyperbolicEmbedding, sr_color)
plt.show()
