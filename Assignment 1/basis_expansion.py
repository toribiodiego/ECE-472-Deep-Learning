#!/usr/bin/env python3.9

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Generate data
def generate_data(num_samples=50):
    X = tf.random.uniform(shape=(num_samples, 1), minval=0, maxval=1)
    y_clean = tf.sin(2 * np.pi * X)
    y = y_clean + tf.random.normal(shape=(num_samples, 1), stddev=0.1)
    return X, y, y_clean


# Compute Gaussian Basis Function
def gaussian_basis_function(x, mu, sigma):
    if sigma == 0:
        return tf.where(x == mu, tf.constant(float("inf"), dtype=tf.float32), 0.0)
    return tf.exp(-tf.square(x - mu) / (sigma**2))


def initialize_gaussian_parameters(X, num_gaussians=10):
    mu = tf.Variable(
        tf.linspace(tf.reduce_min(X), tf.reduce_max(X), num_gaussians), trainable=True
    )
    s = tf.Variable(tf.constant(0.1, shape=[num_gaussians]), trainable=True)
    return mu, s


def xfrm_w_gaussian(x, mus, sigma):
    xfrmed_data = []
    for m, s in zip(mus, sigma):
        xfrmed_data.append(gaussian_basis_function(x, m, s))
    return tf.concat(xfrmed_data, axis=1)


def plot_noisey_sine(X, y_clean, y):
    # Sort X and get the indices
    sorted_indices = tf.argsort(X[:, 0]).numpy()

    # Use the indices to reorder X, y_clean, and y
    X_sorted = tf.gather(X, sorted_indices).numpy()
    y_clean_sorted = tf.gather(y_clean, sorted_indices).numpy()
    y_sorted = tf.gather(y, sorted_indices).numpy()

    plt.figure(figsize=(10, 6))

    # Plot the clean sine wave using a continuous line
    plt.plot(
        X_sorted, y_sorted, "b-", label="Clean Sine Wave"
    )  # Clean sine wave in blue

    # Scatter plot for the noisy data
    plt.scatter(
        X_sorted, y_clean_sorted, color="green", marker="o", label="Noisy Data"
    )  # Noisy data in green

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.ylim(-1.5, 1.5)
    plt.title("Clean Vs Noisy Sine Wave")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def plot_basis_functions(X, xfrmed_data, mu):
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(xfrmed_data):
        plt.plot(X, data, label=f"Gaussian {i+1} (mu={mu[i].numpy():.2f})")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.ylim(0, 1.0)
    plt.title("Transformed Data using Gaussian Basis Function")
    plt.xlabel("X")
    plt.ylabel("Transformed Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y, y_clean = generate_data()
    plot_noisey_sine(X, y, y_clean)
