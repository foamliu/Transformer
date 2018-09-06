from __future__ import print_function

from math import log

import keras
import keras.backend as K
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input, CuDNNLSTM, Bidirectional, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.models import Model

from config import batch_size, num_train_samples, num_valid_samples, vocab_size_zh, embedding_size, Tx
from data_generator import DataGenSequence


def data():
    return DataGenSequence('train'), DataGenSequence('valid')


def create_model():
    input_tensor = Input(shape=(Tx, embedding_size), dtype='float32')
    x = Bidirectional(CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True))(input_tensor)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True)(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = CuDNNLSTM({{choice([256, 512, 1024])}}, return_sequences=True)(x)
    x = Dropout({{uniform(0, 1)}})(x)
    x = TimeDistributed(Dense(vocab_size_zh, activation='softmax'))(x)
    output = x
    model = Model(inputs=input_tensor, outputs=output)

    adam = keras.optimizers.Adam(lr={{loguniform(log(1e-4), log(1))}})
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    model.fit_generator(
        DataGenSequence('train'),
        steps_per_epoch=num_train_samples / batch_size // 50,
        validation_data=DataGenSequence('valid'),
        validation_steps=num_valid_samples / batch_size // 50)

    score, acc = model.evaluate_generator(DataGenSequence('valid'))
    print('Test accuracy:', acc)
    K.clear_session()
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate_generator(DataGenSequence('valid')))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
