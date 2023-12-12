from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, UpSampling2D, GlobalMaxPooling2D
from keras.applications import MobileNet

def create_model():
    """
    Se crea el modelo para entrenar con el CIFAR100 a partir de la transferencia de 
    aprendizaje del modelo preentrenado MobileNet.
    """

    # Descargar modelo MobileNet preentrenado
    IMG_SHAPE = (224, 224, 3)
    mobilenet = MobileNet(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
        dropout=0.25
    )

    mobilenet.trainable = True

    # Adaptar modelo para el conjunto de datos CIFAR100
    model = Sequential(
        [
            UpSampling2D(size=(7, 7),interpolation='bilinear'),
            mobilenet,
            GlobalMaxPooling2D(),
            Dense(100, activation="softmax")
        ]
    )

    return model

mobilenet = create_model()