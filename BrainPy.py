import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

vocab_size = 10000
max_sequence_length = 20

# Inicialización del modelo
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_sequence_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

# Compilación del modelo
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def capture_input():
    # Captura de entrada (audio, imagen, texto, o combinación)
    input_data = input("Input: ")
    return input_data

def generate_text(predictions):
    # Interpretación de las predicciones y generación de respuesta
    max_index = np.argmax(predictions[0])
    generated_text = Tokenizer.index_word[max_index]
    return generated_text

def output_response(generated_text):
    # Salida de la respuesta generada
    print("Output:", generated_text)

def get_user_feedback():
    # Retroalimentación del usuario
    target_data = input("Target: ")
    return target_data

# Bucle continuo de interacción
while True:
    # Captura de entrada (audio, imagen, texto, o combinación)
    input_data = capture_input()

    # Procesamiento y adaptación del modelo a la nueva entrada
    input_sequences = Tokenizer.texts_to_sequences([input_data])
    padded_input = pad_sequences(input_sequences, maxlen=max_sequence_length)

    # Predicción del modelo
    predictions = model.predict(padded_input)

    # Interpretación de las predicciones y generación de respuesta
    generated_text = generate_text(predictions)

    # Salida de la respuesta generada
    output_response(generated_text)

    # Retroalimentación del usuario
    target_data = get_user_feedback()

    # Adaptación continua del modelo con nueva entrada
    model.fit(padded_input, target_data, epochs=1, batch_size=1)

    # Espera antes de la siguiente iteración
    time.sleep(2)


    
