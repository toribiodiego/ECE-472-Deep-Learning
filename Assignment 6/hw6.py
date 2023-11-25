import tensorflow as tf
import numpy as np


class AdamOptimizer:
    def __init__(
        self, variables, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    ):
        self.variables = variables
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in variables
        ]
        self.v = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in variables
        ]
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)


    def apply_gradients(self, grads):
        self.t.assign_add(1)
        lr_t = self.learning_rate * (
            tf.sqrt(1 - tf.pow(self.beta_2, self.t)) / (1 - tf.pow(self.beta_1, self.t))
        )

        for i, (m, v, var, grad) in enumerate(zip(self.m, self.v, self.variables, grads)):
            grad = tf.cast(grad, tf.float32)
            m_t = self.beta_1 * m + (1 - self.beta_1) * grad
            v_t = self.beta_2 * v + (1 - self.beta_2) * tf.square(grad)
            m.assign(m_t)
            v.assign(v_t)
            m_hat = m / (1 - self.beta_1 ** self.t)
            v_hat = v / (1 - self.beta_2 ** self.t)
            var.assign_sub(self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon))


class SyntheticDataset:
    def __init__(self, sentences, vocab_size, seq_len, batch_size):
        self.sentences = sentences
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = self.create_tokenizer()
        self.dataset = self.create_dataset()

    def create_tokenizer(self):
        tokenizer = {
            word: tok
            for tok, word in enumerate(set(" ".join(self.sentences).split()))
        }
        tokenizer["<pad>"] = len(tokenizer)
        tokenizer["<start>"] = len(tokenizer)
        tokenizer["<eos>"] = len(tokenizer)
        return tokenizer

    def create_dataset(self):
        tokenized_sentences = [
            [self.tokenizer["<start>"]]
            + [self.tokenizer[word] for word in sentence.split()]
            + [self.tokenizer["<eos>"]]
            for sentence in self.sentences
        ]

        max_length = max(len(sentence) for sentence in tokenized_sentences)

        padded_sequences = np.array(
            [
                np.pad(
                    sentence,
                    (0, self.seq_len - len(sentence)),
                    "constant",
                    constant_values=self.tokenizer["<pad>"],
                )
                for sentence in tokenized_sentences
            ]
        )

        dataset = tf.data.Dataset.from_tensor_slices(padded_sequences).batch(self.batch_size)
        return dataset

    def reverse_tokenizer(self):
        return {v: k for k, v in self.tokenizer.items()}


class Linear(tf.Module):
    def __init__(self, input_dim, output_dim, name=None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim

        initializer = tf.random_normal_initializer()
        self.w = tf.Variable(
            initializer(shape=(input_dim, output_dim), dtype=tf.float32),
            trainable=True,
            name='weights',
        )
        self.b = tf.Variable(
            tf.zeros(shape=(output_dim,), dtype=tf.float32),
            trainable=True,
            name='biases',
        )

    def __call__(self, x):
        # Perform the linear transformation
        return x @ self.w + self.b


class MLP(tf.Module):
    def __init__(self, dff, d_model):
        super().__init__()
        self.dense1 = Linear(dff, d_model)
        self.dense2 = Linear(d_model, dff)

    def __call__(self, x):
        x = tf.nn.relu(self.dense1(x))
        return self.dense2(x)


class PositionalEncoding(tf.Module):
    def __init__(self, d_model, max_len=5000, name=None):
        super().__init__(name=name)
        # Create a long enough `position_encoding` matrix
        self.position_encoding = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model)
            for pos in range(max_len)
        ])
        # Apply the sine function to even indices (2i)
        self.position_encoding[1:, 0::2] = np.sin(self.position_encoding[1:, 0::2])
        # Apply the cosine function to odd indices (2i+1)
        self.position_encoding[1:, 1::2] = np.cos(self.position_encoding[1:, 1::2])

        # Convert to a tensor
        self.position_encoding = tf.convert_to_tensor(self.position_encoding, dtype=tf.float32)

    def __call__(self, seq_len):
        # Returns the positional encoding for the given sequence length
        return self.position_encoding[:seq_len, :]


class FeedForwardNetwork(tf.Module):
    def __init__(self, d_model, d_ff, dropout_rate, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Initialize the two Linear layers according to the dimensions provided
        self.l1 = Linear(input_dim=d_model, output_dim=d_ff)
        self.l2 = Linear(input_dim=d_ff, output_dim=d_model)

    def __call__(self, x, training):
        # Project up to d_ff
        a = tf.nn.gelu(self.l1(x))
        # Apply dropout using the custom dropout function
        a = dropout(a, rate=self.dropout_rate, training=training)
        # Project back down to d_model
        x = self.l2(a)
        return x


class MultiHeadedAttention(tf.Module):
    def __init__(self, d_model, num_heads, dropout_rate, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_head = d_model // num_heads
        self.dropout_rate = dropout_rate

        self.linear_layers = [
            Linear(d_model, d_model) for _ in range(4)
        ]

    def scaled_dot_product_attention(self, q, k, v, mask, training):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1
        )

        attention_weights = dropout(
            attention_weights, rate=self.dropout_rate, training=training
        )

        output = tf.matmul(attention_weights, v)
        return output

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, q, k, v, mask, training):
        batch_size = tf.shape(q)[0]
        
        q = self.linear_layers[0](q)
        k = self.linear_layers[1](k)
        v = self.linear_layers[2](v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attention_output = self.scaled_dot_product_attention(q, k, v, mask, training)
        
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        output = self.linear_layers[3](attention_output)
        
        return output


class TransformerBlock(tf.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, name=None):
        super().__init__(name=name)
        self.mha = MultiHeadedAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        
        # Initialize gamma and beta as trainable variables for layer normalization
        self.gamma1 = tf.Variable(tf.ones([d_model]), trainable=True, name='gamma1')
        self.beta1 = tf.Variable(tf.zeros([d_model]), trainable=True, name='beta1')
        self.gamma2 = tf.Variable(tf.ones([d_model]), trainable=True, name='gamma2')
        self.beta2 = tf.Variable(tf.zeros([d_model]), trainable=True, name='beta2')

    def __call__(self, x, mask, training):
        # Multi-head self-attention
        attn_output = self.mha(x, x, x, mask, training)
        attn_output = dropout(attn_output, rate=self.mha.dropout_rate, training=training)
        out1 = layer_norm(x + attn_output, self.gamma1, self.beta1)

        # Feedforward network
        ffn_output = self.ffn(out1, training)
        ffn_output = dropout(ffn_output, rate=self.ffn.dropout_rate, training=training)
        out2 = layer_norm(out1 + ffn_output, self.gamma2, self.beta2)

        return out2


class TransformerModel(tf.Module):
    def __init__(
        self, vocab_size, d_model, n_blocks, n_head, d_ff, dropout_rate, name=None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_blocks = n_blocks

        # Token and position embeddings
        self.wte = tf.Variable(
            tf.random.normal([vocab_size, d_model]), name='token_embedding'
        )
        self.wpe = tf.Variable(tf.random.normal([d_model]), name='position_embedding')

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_head, d_ff, dropout_rate)
            for _ in range(n_blocks)
        ]

        # Layer normalization parameters
        self.ln_f_gamma = tf.Variable(
            tf.ones([d_model]), trainable=True, name='ln_f_gamma'
        )
        self.ln_f_beta = tf.Variable(
            tf.zeros([d_model]), trainable=True, name='ln_f_beta'
        )

    def __call__(self, inputs, mask, training):
        # Token + positional embeddings
        x = tf.nn.embedding_lookup(self.wte, inputs) + self.wpe

        # Forward pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask, training)

        # Final layer normalization
        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)

        # Projection to vocab
        logits = tf.matmul(x, self.wte, transpose_b=True)
        return logits


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # normalize x to have mean=0 and var=1 over last axis
    x = (x - mean) / np.sqrt(variance + eps)
    # scale and offset with gamma/beta params
    return g * x + b

def dropout(x, rate, training):
    return tf.nn.dropout(x, rate=rate) if training else x

def train_transformer(dataset, model, epochs=1):
    optimizer = AdamOptimizer(variables=model.trainable_variables)

    for epoch in range(epochs):
        for step, seq in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(seq[:, :-1], mask=None, training=True)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=seq[:, 1:], logits=logits
                    )
                )

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(gradients)

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy()}")

def autoregressive_inference(model, tokenizer, input_sequence, max_length):
    # Generate reverse tokenizer dictionary
    reverse_tok = {v: k for k, v in tokenizer.items()}

    # Tokenize the input sequence
    input_tokens = [
        tokenizer.get(token, tokenizer['<pad>']) for token in input_sequence.split()
    ]
    input_tokens = [tokenizer['<start>']] + input_tokens
    input_tokens = tf.convert_to_tensor(input_tokens, dtype=tf.int32)[None, :]  # Add batch dimension

    # Start generating tokens
    for i in range(max_length):
        logits = model(input_tokens, mask=None, training=False)
        next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
        if next_token.numpy()[0] == tokenizer['<eos>']:
            input_tokens = tf.concat([input_tokens, next_token[:, None]], axis=-1)
            break
        input_tokens = tf.concat([input_tokens, next_token[:, None]], axis=-1)

        print(f"Step {i}: Next token predicted: {reverse_tok[next_token.numpy()[0]]}")

    # Convert token ids back to words
    generated_sequence = [reverse_tok[tok.numpy()] for tok in input_tokens[0]]
    generated_sequence = generated_sequence[1:]  # remove the <start> token
    try:
        eos_index = generated_sequence.index('<eos>')
        generated_sequence = generated_sequence[:eos_index]  # stop at <eos> token
    except ValueError:
        pass  # if no <eos> token is found, pass
    return ' '.join(generated_sequence)

def main():
    # Define parameters
    sentences = ['dog bites man', 'cat drinks milk', 'man eats food']
    vocab_size = 15  # Size of the vocabulary
    seq_len = 5  # Length of the sequence
    batch_size = 3
    d_model = 512
    n_blocks = 2
    n_head = 8
    d_ff = 2048
    dropout_rate = 0.1
    epochs = 200

    # Create dataset
    dataset = SyntheticDataset(sentences, vocab_size, seq_len, batch_size)

    # Create and train the Transformer model
    transformer_model = TransformerModel(
        vocab_size, d_model, n_blocks, n_head, d_ff, dropout_rate
    )
    train_transformer(dataset.dataset, transformer_model, epochs=epochs)

    # Perform autoregressive inference
    start_sentence = 'man eats'
    predicted_sentence = autoregressive_inference(
        transformer_model, dataset.tokenizer, start_sentence, max_length=seq_len
    )
    print(f"Input: {start_sentence}")
    print(f"Predicted continuation: {predicted_sentence}")


if __name__ == "__main__":
    main()
