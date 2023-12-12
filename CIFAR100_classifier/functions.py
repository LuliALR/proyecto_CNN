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
