"""
Este archivo contiene diferentes funciones necesarias para la implementación del clasificador CIFAR-10, 
que incluyen:

    -Carga y preprocesado de los conjuntos de datos.
    -Definición de las arquitecturas de los modelos que se utilizarán
    -Visualización del mapa de características de una CNN.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

def normalize(dataset: np.ndarray) -> np.ndarray:
    """
    Recibe una imagen o un conjunto de imágenes y se encarga de devolver la data normalizada.
    """
    normalize_dataset = dataset / 255.0
    return normalize_dataset


def encode(labelset: np.ndarray, classes: int) -> np.ndarray:
    """
    Recibe un conjunto de estiquetas de las imágenes y las codifica para que sean
    etiquetas de un solo paso en base al número de clases ingresadas, es decir:

    labelset: [6]
    classes: 10
    encode(labelset, classes) -> [[0 0 0 0 0 0 1 0 0 0]]

    """
    labelset = labelset.flatten()
    labelset = tf.one_hot(labelset.astype(np.int32), depth=classes)
    return labelset

def predict(image_filepath: str) -> str:
    """
    Predicción que da el modelo a la imagen indicada por la ubicación del archivo.
    """
    # Load image, resize and normilize
    image = Image.open(image_filepath)
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    
    # Load model
    model = load_model("trained_models/mobilenet_best.h5")
    
    # Classes list
    map = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "cra",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
        ]
    
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
    prediction = prediction[0,:]
    index = np.argmax(prediction)
    label = map[index]
    
    # Make lists with labels and their probabilities of the first five predictions
    labels = []
    probabilities = []
    for i in range (5):
        index = np.argmax(prediction)
        labels.append(map[index])
        probabilities.append(prediction[index])
        del map[index]
        prediction = np.delete(prediction, index)

    return label, labels, probabilities

def map_features(original_model, activation_model, layer: int, image: np.ndarray):
    """
    Muestra el mapa de características de la imagen de entrada al hacerla pasar por el modelo de
    activación, que es el modelo parcial de la arquitectura completa hasta la capa indicada.

    """
    activation_model = tf.keras.models.Model(
        inputs=activation_model.inputs, outputs=original_model.layers[layer].output
    )
    activation = activation_model(np.expand_dims(image, axis=0))

    fig = plt.figure()
    fig.suptitle(original_model.layers[layer].__class__.__name__)
    for i in range(activation.shape[-1]):
        plt.subplot(4, int(activation.shape[-1] / 4), i + 1)
        plt.axis("off")
        plt.imshow(activation[0, :, :, i])
    plt.show()
    return activation_model
