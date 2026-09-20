"""
perceptron.py - single-layer perceptron implemented from scratch.

Shared, UNCHANGED, between Exercise 1 (separable data) and Exercise 2
(overlapping data + pocket algorithm). No third-party model (no
sklearn.linear_model.Perceptron, no SGDClassifier) is used anywhere.
"""
import numpy as np


def step(z):
    """Step activation: 1 if z >= 0 else 0."""
    return np.where(z >= 0, 1, 0)


def predict(w, b, X):
    """Vectorized prediction y_hat = step(w . x + b) for a batch X (n, d)."""
    z = X @ w + b
    return step(z)


def accuracy(w, b, X, y):
    """Fraction of samples correctly classified."""
    return float(np.mean(predict(w, b, X) == y))


def init_weights(n_features, rng):
    """
    Non-zero initialization (item B): w ~ N(0, 0.01), b = 0.

    Starting at w = 0 would make the learning rate provably irrelevant
    (see Exercise 1, item D3) -- this is why a tiny random w is used instead.
    """
    w = rng.normal(0.0, 0.01, size=n_features)
    b = 0.0
    return w, b


def train(X, y, w0, b0, eta, max_epochs=100, on_update=None):
    """
    Trains a perceptron with the {0,1} error-driven update rule:

        y_hat = step(w . x + b)
        w <- w + eta * (y - y_hat) * x
        b <- b + eta * (y - y_hat)

    Starts from the given (w0, b0) so that two runs with different eta can
    be compared with everything else held fixed. Stops when a full pass
    over the data produces zero updates, or after max_epochs.

    on_update(w, b, epoch, acc), if given, is called after EVERY individual
    weight update (not just once per epoch) -- Exercise 2 uses this hook to
    track the pocket (best-so-far) weights without touching this loop.

    Returns: w, b, epoch_accuracy (list), epoch_updates (list), n_epochs
    """
    w, b = w0.copy(), float(b0)
    n = X.shape[0]
    epoch_accuracy = []
    epoch_updates = []
    epoch = 0

    for epoch in range(max_epochs):
        n_updates = 0
        for i in range(n):
            xi = X[i]
            yi = y[i]
            y_hat = int(step(xi @ w + b))
            error = yi - y_hat
            if error != 0:
                w = w + eta * error * xi
                b = b + eta * error
                n_updates += 1
                if on_update is not None:
                    on_update(w, b, epoch, accuracy(w, b, X, y))
        epoch_accuracy.append(accuracy(w, b, X, y))
        epoch_updates.append(n_updates)
        if n_updates == 0:
            break

    n_epochs = epoch + 1
    return w, b, epoch_accuracy, epoch_updates, n_epochs
