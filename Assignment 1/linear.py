#!/bin/env python
import matplotlib.pyplot as plt
import tensorflow as tf
from basis_expansion import (generate_data, initialize_gaussian_parameters,
                             plot_basis_functions, xfrm_w_gaussian)


class Linear(tf.Module):
    #
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def plot_regression_data(X, y, y_clean, y_hat):
    plt.figure(figsize=(10, 6))

    # Plot the clean sine wave
    plt.plot(
        X.numpy().squeeze(), y_clean.numpy().squeeze(), "b-", label="Clean Sine Wave"
    )

    # Plot the noisy data
    plt.scatter(
        X.numpy().squeeze(),
        y.numpy().squeeze(),
        color="green",
        marker="o",
        label="Noisy Data",
    )

    # Plot the model's predictions
    plt.plot(
        X.numpy().squeeze(), y_hat.numpy().squeeze(), "r--", label="Model Predictions"
    )

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.ylim(-1.5, 1.5)
    plt.title("Regression Results")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import yaml
    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument(
        "-c", "--config", type=Path, default=Path("config_noisy_sine.yaml")
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    num_gaussians = config["model"]["num_gaussians"]
    num_inputs = num_gaussians
    num_outputs = 1

    x, y, y_clean = generate_data(num_samples=num_samples)
    mu, s = initialize_gaussian_parameters(x, num_gaussians)

    linear = Linear(num_inputs, num_outputs)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=num_samples, dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            x_batch_transformed = xfrm_w_gaussian(x_batch, mu, s)

            y_hat = linear(x_batch_transformed)
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(loss, list(linear.trainable_variables) + [mu, s])
        grad_update(step_size, list(linear.trainable_variables) + [mu, s], grads)
        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

        x_transformed = xfrm_w_gaussian(x, mu, s)
        y_hat = linear(x_transformed)

        # Sort X and get the indices
        sorted_indices = tf.argsort(x[:, 0]).numpy()

        # Use the indices to reorder X, y, y_clean, and y_hat
        x = tf.gather(x, sorted_indices)
        y = tf.gather(y, sorted_indices)
        y_clean = tf.gather(y_clean, sorted_indices)
        y_hat = tf.gather(y_hat, sorted_indices)

        plot_regression_data(x, y, y_clean, y_hat)

        # After training the model and before plotting the regression data
        x_transformed_for_plotting = xfrm_w_gaussian(x, mu, s)
        plot_basis_functions(
            x.numpy().squeeze(), x_transformed_for_plotting.numpy().T, mu
        )
