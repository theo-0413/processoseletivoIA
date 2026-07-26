# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

import collections
import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist

# 1. Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Train-Validation Split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, stratify=y_train, test_size=0.25, random_state=42
)

print('Quantidade de imagens de treino:', x_train.shape[0])
print('Quantidade de imagens de validação:', x_val.shape[0])
print('Quantidade de imagens de teste:', x_test.shape[0])

# 3. Plotting Class Distribution (Fixed overlapping bars)
counterTrain = collections.Counter(y_train)
counterVal = collections.Counter(y_val)
counterTest = collections.Counter(y_test)

fig, ax = pyplot.subplots(figsize=(8, 4))
x = np.arange(10)
width = 0.25

ax.bar(x - width, [counterTrain[i] for i in range(10)], width, label='Treino')
ax.bar(x, [counterVal[i] for i in range(10)], width, label='Validação')
ax.bar(x + width, [counterTest[i] for i in range(10)], width, label='Teste')

ax.set_title('Imagens por dígito')
ax.set_ylabel('Quantidade de imagens')
ax.set_xlabel('Dígito')
ax.set_xticks(x)
ax.legend()
pyplot.tight_layout()
pyplot.show()

# 4. Reshape & Normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 5. Build Model Architecture (Fixed depth & input_shape issues)
model = models.Sequential([
    # Block 1
    layers.Conv2D(28, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    layers.Conv2D(56, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

# 6. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 1. Define Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# 2. Define Model Checkpoint (Saves best .h5 file automatically)
checkpoint = ModelCheckpoint(
    filepath='model.h5', # Output filename
    monitor='val_loss',                 # Metric to monitor
    save_best_only=True,                # Only overwrite if val_loss improves
    mode='min',                         # Minimize val_loss
    verbose=1
)

# 3. Train with both callbacks
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, checkpoint]  # Pass both callbacks here
)

# Avaliando a CNN treinada
score = model.evaluate(x_test, y_test)

print( '\nPerda:{:.3f}\nAcurácia:{}'.format( score[0], score[1] ) )

# Imprimindo uma imagem de exemplo
image_index = 2222
pyplot.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

# Predizendo o dígito dessa imagem
pred = model.predict( x_test[image_index].reshape(1, 28, 28, 1) )
print( '\nO valor predito é:', pred.argmax() )

# Histórico de acurácia
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('Acurácia do modelo no treino e validação')
pyplot.ylabel('Acurácia')
pyplot.xlabel('Época')
pyplot.legend(['Treino', 'Validação'], loc='upper left')
pyplot.show()

# Histórico da função de perda
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('Perda do modelo no treino e validação')
pyplot.ylabel('Perda')
pyplot.xlabel('Época')
pyplot.legend(['Treino', 'Validação'], loc='upper left')
pyplot.show()