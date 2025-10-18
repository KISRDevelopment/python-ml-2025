import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.datasets as gen
import numpy.random as rng

# Generate simple binary classification dataset
X, y = gen.make_blobs(n_samples=1000, centers=4, n_features=2, random_state=31, cluster_std=4)
X /= 20
y = y % 3
y = (y == 1) | (y == 2)

class0_ix = y == 0
class1_ix = y == 1

# plot dataset
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(X[class0_ix, 0], X[class0_ix, 1])
ax.scatter(X[class1_ix, 0], X[class1_ix, 1], color='orange')
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
plt.savefig('tmp/block11_nonseparable_binary_classification.svg', pad_inches=0.05, bbox_inches='tight')

# plot different classifier lines
candidate_lines = [
    [0.5, 0.5, 0],
    [1, 0, 0],
    [-1, 1, 0]
]

f, axes = plt.subplots(1, 3, figsize=(29, 8))

for i, l in enumerate(candidate_lines):
    a, b, c = l
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 1000)
    if b != 0:
        y = (c - a * x1) / b 
    else:
        x1 = np.ones_like(x1) * c / a
        y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 1000)
    
    ax = axes[i]
    ax.plot(x1, y, color='magenta', linewidth=5, linestyle='dashed')

    # plug points into the lines equation
    yhat = a * X[:, 0] + b * X[:, 1]

    # compute diff
    diff = yhat - c

    # generate predictions
    pred_class1 = diff < 0

    # plot predictions
    ax.scatter(X[~pred_class1, 0], X[~pred_class1, 1])
    ax.scatter(X[pred_class1, 0], X[pred_class1, 1], color='orange')
    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
plt.savefig('tmp/block11_nonseparable_binary_classification_lines.svg', pad_inches=0.05, bbox_inches='tight')

#
# Activation functions
#
fig, axes = plt.subplots(1, 3, figsize=(15,3))
xs = np.linspace(-5, 5, 100)
funcs = {
    'tanh' : np.tanh, 
    'relu' : lambda x: np.maximum(0, x),
    'sigmoid' : lambda x: 1/(1+np.exp(-x))
}
for i, func_name in enumerate(funcs.keys()):
    func = funcs[func_name]
    ys = func(xs)
    ax = axes[i]
    ax.plot(xs, ys, linewidth=2, label=func_name)
    ax.grid(True, linestyle='--')
    ax.set_xlabel('$x$', fontsize=18)
    ax.set_ylabel('$y$', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_title(func_name, fontsize=20)
fig.subplots_adjust(wspace=0.28)
fig.savefig('tmp/block11_activations.svg', bbox_inches='tight')