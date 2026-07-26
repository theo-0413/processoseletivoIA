import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# 1. Load the trained Keras model
model_path = 'model.h5'
print(f'Carregando o modelo de: {model_path}')
model = tf.keras.models.load_model(model_path)

# 2. Initialize the TFLite Converter from the loaded model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Apply Optimization (Dynamic Range Quantization)
# This shrinks model size (usually by ~4x) and speeds up CPU inference 
# by quantizing weights from float32 to int8 on-the-fly.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Convert the model
print('Convertendo o modelo para TensorFlow Lite com quantização...')
tflite_model = converter.convert()

# 5. Save the optimized model to disk
tflite_output_path = 'model.tflite'
with open(tflite_output_path, 'wb') as f:
    f.write(tflite_model)

print(f'Modelo otimizado salvo com sucesso em: {tflite_output_path}')