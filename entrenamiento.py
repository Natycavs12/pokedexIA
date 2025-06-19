import tensorflow as tf
from tensorflow import keras
# from keras.preprocessing import image_dataset_from_directory
# from keras.applications import MobileNetV2
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.models import Model
# from keras.callbacks import EarlyStopping
from keras import layers, models, callbacks
# from tensorflow.keras.applications import MobileNetV2
from tensorflow import MobileNetV2

image_dataset_from_directory = tf.keras.utils.image_dataset_from_directory

# --- 1. Cargar dataset ---

DATASET_PATH = 'C:/Users/natycavs/Desktop/pokemon/pokeapp/PokemonData/'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Clases detectadas: {class_names}")

# --- 2. Preparar modelo preentrenado ---

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False,
                         weights='imagenet')

# Congelamos el modelo base para transfer learning
base_model.trainable = False

# Creamos el modelo completo
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Entrenar el modelo ---

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stop]
)

# --- 4. Guardar modelo ---

model.save('pokemon_classifier.h5')
print("Modelo guardado como 'pokemon_classifier.h5'")
