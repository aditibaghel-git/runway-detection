from functools import partial
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, Conv2DTranspose,
    Flatten, Dense
)
from tensorflow.keras.models import Model

DefaultConv2D = partial(
    Conv2D,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer="he_normal"
)

def build_model(input_shape, coord_vector_size):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 3, strides=2, padding="same", activation="relu")(inputs)
    x = MaxPool2D(2)(x)

    x = DefaultConv2D(128)(x)
    x = DefaultConv2D(128)(x)
    x = MaxPool2D(2)(x)

    x = DefaultConv2D(256)(x)
    shared = DefaultConv2D(256)(x)

    # Mask head
    m = Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(shared)
    m = Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(m)
    m = Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(m)
    mask_output = Conv2D(1, 1, activation="sigmoid", name="mask_output")(m)

    # Anchor head
    a = Flatten()(shared)
    a = Dense(128, activation="relu")(a)
    a = Dense(64, activation="relu")(a)
    anchor_output = Dense(coord_vector_size, activation="sigmoid", name="anchor_output")(a)

    return Model(inputs, [mask_output, anchor_output])