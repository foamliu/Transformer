import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, CuDNNLSTM, Bidirectional, TimeDistributed
from keras.models import Model
from keras.utils import plot_model

from config import hidden_size, vocab_size_zh, embedding_size, Tx


def build_model():
    input_tensor = Input(shape=(Tx, embedding_size), dtype='float32')
    x = Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True))(input_tensor)
    x = CuDNNLSTM(hidden_size, return_sequences=True)(x)
    x = CuDNNLSTM(256, return_sequences=True)(x)
    x = TimeDistributed(Dense(vocab_size_zh, activation='softmax'))(x)
    outputs = x
    model = Model(inputs=input_tensor, outputs=outputs)
    return model


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)

    K.clear_session()
