import argparse 
import tensorflow as tf
import os
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path



class Linear(tf.Module):
    def __init__(self, input_dim, output_dim, bias=True, name=None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim

        initializer = tf.random_normal_initializer()
        self.w = tf.Variable(
            initializer(shape=(input_dim, output_dim), dtype=tf.float32),
            trainable=True,
            name='weights',
        )
        self.b = None
        if bias:
            self.b = tf.Variable(
                tf.zeros(shape=(output_dim,), dtype=tf.float32),
                trainable=True,
                name='biases',
            )

    def __call__(self, x):
        output = x @ self.w
        if self.b is not None:
            output += self.b
        return output


# SineInitalizer and FirstLayerSineInitalizer seem to have different 
# limits, but they both seem to generate uniformly and using the same type of 'scheme'
class SineInitializer(tf.Module):
    def __init__(self, omega):
        super(SineInitializer, self).__init__()
        self.omega = omega

    def __call__(self, shape, dtype=tf.float32):
        # Calculate the limit as per the "Glorot" initialization scheme
        limit = tf.sqrt(6 / shape[0]) / self.omega
        # Generate a random tensor with values uniformly distributed between -limit and limit
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)


class FirstLayerSineInitializer(tf.Module):
    def __call__(self, shape, dtype=tf.float32):
        # The limit is set as 1 divided by the first dimension of the shape
        limit = 1 / shape[0]
        # Initialize and return a tensor with values uniformly distributed between -limit and limit
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)


class SineLayer(tf.Module):
    def __init__(self, input_dim, out_features, bias=True, omega_0=30, initializer=None, name=None):
        super().__init__(name=name)
        self.omega_0 = omega_0
        self.initializer = initializer if initializer is not None else tf.random_normal_initializer()
        self.linear = Linear(input_dim, out_features, bias=bias)  # Replace Keras Dense with Linear
        self.init_weights()

    def init_weights(self):
        # Initialize weights using the provided initializer
        self.linear.w.assign(self.initializer(self.linear.w.shape, self.linear.w.dtype))
        if self.linear.b is not None:
            self.linear.b.assign(tf.zeros(self.linear.b.shape, self.linear.b.dtype))
        
    def __call__(self, input):
        return tf.sin(self.omega_0 * self.linear(input))

    def call_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return tf.sin(intermediate), intermediate


class Siren(tf.Module):
    def __init__(self, input_dim, units, num_layers, out_features, outermost_linear=False, hidden_omega=30.0):
        super().__init__()
        self.layers = []
        
        # First layer with a specific initializer
        self.layers.append(SineLayer(input_dim, units, omega_0=hidden_omega, initializer=FirstLayerSineInitializer()))
        
        # Hidden layers with SineLayer
        for _ in range(num_layers - 1):
            self.layers.append(SineLayer(units, units, omega_0=hidden_omega, initializer=SineInitializer(hidden_omega)))
        
        # Output layer
        if outermost_linear:
            self.layers.append(Linear(units, out_features, bias=True))  # Assumes Linear has no initializer
        else:
            self.layers.append(SineLayer(units, out_features, omega_0=hidden_omega, initializer=SineInitializer(hidden_omega)))
    
    def __call__(self, coords):
        x = coords
        for layer in self.layers:
            x = layer(x)
        return x



def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [tf.linspace(-1, 1, num=sidelen)])
    mgrid = tf.stack(tf.meshgrid(*tensors, indexing="ij"), axis=-1)
    mgrid = tf.reshape(mgrid, [-1, dim])
    return mgrid


def get_img(img_path, img_size):
    img_raw = tf.io.read_file(img_path)
    img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)
    img_ground_truth = tf.image.resize(img_ground_truth, [img_size, img_size])

    mgrid = get_mgrid(img_size, 2)
    mgrid = tf.cast(mgrid, tf.float32)
    return (
        mgrid,
        tf.reshape(img_ground_truth, [img_size * img_size, 3]),
        img_ground_truth,
    )


def train_step(model, optimizer, loss_fn, img_mask, img_train):
    with tf.GradientTape() as tape:
        predictions = model(img_mask)
        loss = loss_fn(img_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_siren(image_path, batch_size, num_epochs, img_size):
    img_mask, img_train, _ = get_img(image_path, img_size=img_size)

    model = Siren(
        input_dim=2,
        units=256,
        num_layers=3,
        out_features=3,
        outermost_linear=True,
        hidden_omega=30.0,
    )

    optimizer = tf.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.losses.MeanSquaredError()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for step in range(0, len(img_mask), batch_size):
            img_mask_batch = img_mask[step:step + batch_size]
            img_train_batch = img_train[step:step + batch_size]
            loss = train_step(model, optimizer, loss_fn, img_mask_batch, img_train_batch)
            epoch_loss += loss.numpy()

        epoch_loss /= (len(img_mask) / batch_size)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    
    return model



def eval(model, test_img_path, output_dir, img_size):
    img_mask, _, img_ground_truth = get_img(test_img_path, img_size=img_size)
    
    predicted_image = model(img_mask).numpy().reshape(img_ground_truth.shape)
    predicted_image = predicted_image.clip(0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_ground_truth.numpy())
    axes[0].set_title("Ground Truth Image")
    axes[0].axis("off")

    axes[1].imshow(predicted_image)
    axes[1].set_title("Predicted Image")
    axes[1].axis("off")

    plt.tight_layout()
    output_image_path = os.path.join(output_dir, f"{Path(test_img_path).stem}_predicted.png")
    plt.savefig(output_image_path, dpi=150, transparent=True)
    plt.show()

    print(f"Evaluated image saved to {output_image_path}")





def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SIREN on an image")

    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image for training/evaluation.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store outputs like models and images.")
    parser.add_argument("--train", action="store_true", help="Flag to train the model")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for the model")

    args = parser.parse_args()

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # GPU configuration (optional)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if args.train:
        print("Starting training...")
        trained_model = train_siren(
            args.input_image,
            args.batch_size,
            args.n_epochs,
            args.img_size
        )
        print("Training finished.")

    print("Starting evaluation...")
    eval(
        trained_model,
        args.input_image,
        args.output_dir,
        args.img_size
    )
    print("Evaluation finished.")


if __name__ == "__main__":
    main()

