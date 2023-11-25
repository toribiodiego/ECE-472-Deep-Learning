import os
import re
import yaml
import string
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

SEED = 23  
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class LogisticRegressionModel(tf.Module):
    def __init__(self, input_dim, num_labels):
        self.weights = tf.Variable(tf.random.normal([input_dim, num_labels]), name='weights')
        self.bias = tf.Variable(tf.zeros([num_labels]), name='bias')

    def __call__(self, inputs):
        return tf.matmul(inputs, self.weights) + self.bias

def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
    return accuracy.numpy()

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)
    gradients = tape.gradient(loss, [model.weights, model.bias])
    optimizer.apply_gradients(zip(gradients, [model.weights, model.bias]))
    return loss

def evaluate_model(model, eval_data, phase, config):
    model_name = config['model']['name']
    sentence_model = SentenceTransformer(model_name)
    
    eval_texts = eval_data['text'].tolist()
    eval_labels = eval_data['label'].to_numpy(dtype='int32') 
    
    eval_embeddings = sentence_model.encode(eval_texts)
    
    logits = model(eval_embeddings)
    loss = compute_loss(logits, eval_labels)
    accuracy = compute_accuracy(logits, eval_labels)
    
    print(f"{phase} Loss: {loss.numpy():.4f}")
    print(f"{phase} Accuracy: {accuracy:.2%}")

def train_model(config, train_data, val_data):
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['epochs']
    learning_rate = float(config['training']['learning_rate'])
    reduction_factor = float(config['training']['reduction_factor'])
    patience = config['training']['patience']
    
    model_name = config['model']['name']
    sentence_model = SentenceTransformer(model_name)
    
    train_texts = train_data['text'].tolist()
    train_labels = train_data['label'].to_numpy(dtype='int32')
    val_texts = val_data['text'].tolist()
    val_labels = val_data['label'].to_numpy(dtype='int32')

    train_embeddings = sentence_model.encode(train_texts)
    val_embeddings = sentence_model.encode(val_texts)

    input_dim = train_embeddings.shape[1]  
    num_labels = config['model']['num_labels']

    model = LogisticRegressionModel(input_dim, num_labels)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training loop
        for i in range(0, len(train_embeddings), batch_size):
            batch_x = train_embeddings[i:i+batch_size]
            batch_y = train_labels[i:i+batch_size]
            train_loss = train_one_step(model, optimizer, batch_x, batch_y)
            
        # Validation loop
        val_logits = model(val_embeddings)
        val_loss = compute_loss(val_logits, val_labels)
        val_accuracy = compute_accuracy(val_logits, val_labels)
        
        # Logging
        train_logits = model(train_embeddings)
        train_accuracy = compute_accuracy(train_logits, train_labels)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss.numpy()}, Accuracy: {train_accuracy:.2%}, Val Loss: {val_loss.numpy()}, Val Accuracy: {val_accuracy:.2%}")
        
        # Learning rate reduction check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Reduce learning rate if patience is exceeded
        if patience_counter >= patience:
            learning_rate *= reduction_factor
            optimizer.lr.assign(learning_rate)
            patience_counter = 0
            print(f"Reducing learning rate to {learning_rate}")
    
    return model

def split_data(data, test_size, val_size, random_state, label_column='label'):
    train_data, temp_data = train_test_split(data, test_size=test_size + val_size, random_state=random_state, stratify=data[label_column])
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=temp_data[label_column])
    return train_data, val_data, test_data

def load_ag_news_data():
    dataset = load_dataset('ag_news')
    train_df = pd.DataFrame(dataset['train'])
    
    return train_df

def preprocess_text_single_column(df, column_name, config):
    # Ensure text data is string
    df[column_name] = df[column_name].astype(str)
    
    # Convert text to lowercase
    df[column_name] = df[column_name].apply(lambda x: x.lower())
    
    if config['preprocessing']['remove_punctuation']
        # Remove punctuation
        pattern_elements = ""
        if config['preprocessing']['remove_punctuation']:
            pattern_elements += string.punctuation
        pattern = rf'[{pattern_elements}]'
        df[column_name] = df[column_name].apply(lambda x: re.sub(pattern, '', x))
    
    return df

def main(config):
    # 1. Load Data
    dataset = load_ag_news_data().sample(n=config['data']['sample_size'], 
                                         random_state=config['data']['random_state'])  
    
    # 2. Preprocess Text
    column_name = 'text'
    preprocessed_dataset = preprocess_text_single_column(dataset, column_name, config)

    # 3. Split Data
    train_data, val_data, test_data = split_data(preprocessed_dataset, 
                                                 config['data']['test_size'], 
                                                 config['data']['val_size'], 
                                                 config['data']['random_state'])
 
    # 4. Train Model
    print("Starting model training...")
    model = train_model(config, train_data, val_data)

    print("Model training completed.")

    # 5. Evaluate on Test Data if enabled
    if config['evaluation']['use_test_data']:
        print("Evaluating on test data...")
        evaluate_model(model, test_data, "Testing", config)

if __name__ == "__main__":
    main(config)
