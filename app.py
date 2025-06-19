########################################################
""" LIBRERIAS """
# importación de liberías
########################################################

import numpy as np
from flask import Flask, request, jsonify, render_template
from config import Config

from PIL import Image
import tensorflow as tf
import io
import csv

########################################################
""" CONFIGURACIÓM """
# configuracion de la app
########################################################

app = Flask(__name__)	
app.config.from_object(Config)

# Cargar el modelo
model = tf.keras.models.load_model('C:/Users/natycavs/Desktop/pokemon/pokeapp/pokemon_classifier.h5')

# Mapeo de clases a nombres de Pokémon (ajustar según tu dataset)
class_names = ['Abra','Aerodactyl','Alakazam','Alolan Sandslash','Arbok','Arcanine','Articuno',
'Beedrill','Bellsprout','Blastoise','Bulbasaur','Butterfree',
'Caterpie','Chansey','Charizard','Charmander','Charmeleon','Clefable','Clefairy','Cloyster','Cubone',
'Dewgong','Digglet','Ditto','Dodrio','Doduo','Dragonair','Dragonite','Dratini','Drowzee','Dugtrio',
'Eevee','Ekans','Electabuzz','Electrode','Exeggcute','Exeggutor',
'Farfetchd','Fearow','Flareon',
'Gastly','Gengar','Geodude','Gloom','Golbat','Goldeen','Golduck','Golem','Graveler','Grimer','Growlithe','Gyarados',
'Haunter','Hitmonchan','Hitmonlee','Horsea','Hypno',
'Ivysaur','Jigglypuff','Jolteon','Jynx',
'Kabuto','Kabutops','Kadabra','Kakuna','Kangaskhan','Kingler','Koffing','Krabby',
'Lapras','Lichitung',
'Machamp','Machoke','Machop','Magikarp','Magmar','Magnemite','Magneton','Mankey','Marowak',
'Meowth','Metapod','Mew','Mewtwo','Moltres','MrMime','Muk',
'Nidoking','Nidoqueen','Nidorina','Nidorino','Ninetales',
'Oddish','Omanyte','Omastar','Onix',
'Paras','Parasect','Persian','Pidgeot','Pidgeotto','Pidgey','Pikachu','Pinsir','Poliwag',
'Poliwhirl','Poliwrath','Ponyta','Porygon','Primeape','Psyduck',
'Raichu','Rapidash','Raticate','Rattata','Rhydon','Rhyhorn',
'Sandshrew','Sandslash','Scyther','Seadra','Seaking','Seel','Shellder',
'Slowbro','Slowpoke','Snorlax','Spearow','Squirtle','Starmie','Staryu',
'Tangela','Tauros','Tentacool','Tentacruel',
'Vaporeon','Venomoth','Venonat','Venusaur','Victreebel','Vileplume','Voltorb','Vulpix',
'Wartortle','Weedle','Weepinbell','Weezing','Wigglypuff',
'Zapdos','Zubat'] 

########################################################
""" FUNCIONES """
########################################################

def leer_csv(pokemon_name):
    csv_path = 'C:/Users/natycavs/Desktop/pokemon/pokeapp/pokemon_lista_pokedex.csv'
    with open(csv_path, 'r') as file:

        reader = csv.reader(file)
        for row in reader:
            if pokemon_name in row:
                return row[2].strip() 

########################################################
""" RUTAS """
########################################################

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/pokedex')
def pokedex():
    return render_template("pokedex.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file'].read()
    image = Image.open(io.BytesIO(file)).convert('RGB')
    image = image.resize((224, 224))
    
    # Preprocesar imagen
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Realizar predicción
    predictions = model.predict(img_array)
    # print("todas las predictions: ",predictions)
    probs = predictions[0]  # shape: (num_classes,)

    # probs: array de probabilidades para cada clase, por ejemplo, salida de model.predict_proba()
    # class_names: lista con los nombres de las clases

    # Obtener los índices de las 3 probabilidades más altas
    top3_indices = np.argsort(probs)[-3:][::-1]
    print("top3_indices: ",top3_indices)

    # Obtener los nombres de las clases y sus probabilidades
    top3_classes = [class_names[i] for i in top3_indices]
    top3_confidences = [float(probs[i]) for i in top3_indices]


    # predicted_class1 = class_names[np.argmax(predictions[0])]
    # predicted_class2 = class_names[np.argmax(predictions[1])]
    # predicted_class3 = class_names[np.argmax(predictions[2])]
    predicted_class1 = top3_classes[0]
    predicted_class2 = top3_classes[1]
    predicted_class3 = top3_classes[2]

    # print("predicted_class: ",predicted_class)
    # confidence1 = float(np.max(predictions[0]))
    # confidence2 = float(np.max(predictions[1]))
    # confidence3 = float(np.max(predictions[2]))

    confidence1 = round(top3_confidences[0], 2)
    confidence2 = round(top3_confidences[1], 2)
    confidence3 = round(top3_confidences[2], 2)

    print("1) pokemon: ", predicted_class1, " confidence: ",confidence1)
    print("2) pokemon: ", predicted_class2, " confidence: ",confidence2)
    print("3) pokemon: ", predicted_class3, " confidence: ",confidence3)

    descripcion1 = leer_csv(predicted_class1)
    # descripcion2 = leer_csv(predicted_class2)
    # descripcion3 = leer_csv(predicted_class3)

    return jsonify({
        'pokemon1': predicted_class1,
        'confidence1': confidence1,
        'descripcion1': descripcion1,
        'pokemon2': predicted_class2,
        'confidence2': confidence2,
        # 'descripcion2': descripcion2,
        'pokemon3': predicted_class3,
        'confidence3': confidence3
        # 'descripcion3': descripcion3
    })

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000, debug=True)
    app.run(port=8000, debug=True)