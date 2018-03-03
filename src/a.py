from keras.models import Model
from keras.layers import Embedding, Dense, Input, Concatenate
from keras.initializers import he_normal


def get_model():
    input_x = Input(shape=(1,))
    x = Embedding(1, 50)(input_x)

    input_y = Input(shape=(1,))
    y = Embedding(1, 50)(input_y)

    merge = Concatenate(axis=-1)([x, y])
    merge = Dense(50, activation="relu", kernel_initializer=he_normal(seed=42))(merge)
    output = Dense(6, activation="sigmoid")(merge)
    model = Model(inputs=[input_x, input_y], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    return model