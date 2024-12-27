import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Load and preprocess the text
filepath = tf.keras.utils.get_file('shakespeare.txt', origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]  # Use a subset of the text

# Create character mappings
characters = sorted(set(text))  # All unique characters
char_to_index = {c: i for i, c in enumerate(characters)}  # Map characters to indices
index_to_char = {i: c for i, c in enumerate(characters)}  # Map indices to characters

# Hyperparameters
SEQ_LENGTH = 40  # Length of each input sequence
STEP_SIZE = 3    # Step size to slide the window over the text

# Prepare input sequences and corresponding outputs
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

# One-hot encoding
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.float32)
y = np.zeros((len(sentences), len(characters)), dtype=np.float32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Build the LSTM model
model = Sequential()

# LSTM layer
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters)), return_sequences=False))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Output layer (Dense with softmax activation for character prediction)
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001))

# Train the model
model.fit(x, y, batch_size=128, epochs=4)

# Save the trained model
model.save('textgenerator.model')

# Sampling function for generating text
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate text function
def generate_text(length, temperature=1.0):
    start_index = np.random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))  # Prepare the input for prediction
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_pred, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# Generate some text with different temperatures
print('-------0.2--------')
print(generate_text(300, 0.2))

print('-------0.4--------')
print(generate_text(300, 0.4))

print('-------0.6--------')
print(generate_text(300, 0.6))

print('-------0.8--------')
print(generate_text(300, 0.8))

print('-------1.0--------')
print(generate_text(300, 1.0))
