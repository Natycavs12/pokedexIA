########################################################
""" LIBRERIAS """
# importación de liberías
########################################################

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers 
# from keras import layers
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
        # lines = file.readlines()
        # Procesar las líneas del CSV según sea necesario
        # for line in lines:
            # print(line.strip()) 
            # if pokemon_name in line:
            #     return line.split(",")[2].strip()

        reader = csv.reader(file)
        for row in reader:
            if pokemon_name in row:
                return row[2].strip() 

########################################################
""" RUTAS """
########################################################
@app.route("/")
def index():
    # entrenar()
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
    print("predictions: ",predictions)
    predicted_class = class_names[np.argmax(predictions[0])]
    # print("predictions 1: ",np.argmax(predictions[0]))
    # print("predictions 1: ",class_names[np.argmax(predictions[2])])

    # print("predicted_class: ",predicted_class)
    confidence = float(np.max(predictions[0]))
    print("pokemon: ", predicted_class, " confidence: ",confidence)
    descripcion = leer_csv(predicted_class)
    # print("descripcion: ", descripcion)

    return jsonify({
        'pokemon': predicted_class,
        'confidence': confidence,
        'descripcion': descripcion
    })


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000, debug=True)
    app.run(port=8000, debug=True)