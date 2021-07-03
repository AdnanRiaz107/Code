from keras.callbacks import Callback
from keras.callbacks import EarlyStopping ,TensorBoard
from keras.layers import *
np.random.seed(1024)
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from time import time
import time

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping ,TensorBoard
from keras.layers import *
np.random.seed(1024)
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from time import time
import time
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class Attention(Layer):
    def __init__(self, step_dim , bias=True,W_regularizer=None, b_regularizer=None,W_constraint=None, b_constraint=None, **kwargs):
        """
                Keras Layer that implements an Attention mechanism for temporal data.
                Supports Masking.
                Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
                # Input shape
                    3D tensor with shape: `(samples, steps, features)`.
                # Output shape
                    2D tensor with shape: `(samples, features)`.
                :param kwargs:
                Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
                The dimensions are inferred based on the output shape of the RNN.
                Example:
                    model.add(LSTM(64, return_sequences=True))
                    model.add(Attention())
                """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))


        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

    def get_config(self):
        # For serialization with 'custom_objects'

        config = super().get_config()
        config['input_shape'] = self.step_dim
        config['bias'] = self.bias
        config['W_regularizer'] = self.W_regularizer
        config['b_regularizer'] = self.b_regularizer
        config['.W_constraint'] = self.W_constraint
        config['b_constraint'] = self.b_constraint
        return config

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc=[]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))



import keras
kullback_leibler_divergence = keras.losses.kullback_leibler_divergence
K = keras.backend
#In this example, 0.01 is the regularization weight, and 0.05 is the sparsity target.
def kl_divergence_regularizer(inputs):
        means = K.mean(inputs, axis=0)
        return 0.01 * (kullback_leibler_divergence(0.05, means)
                       + kullback_leibler_divergence(1 - 0.05, 1 - means))

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
#BDLSTM with FCN
def BDLSTMS(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), activation='sigmoid', output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]),activation='sigmoid', output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(main_output)
   #BDLSTM_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True),merge_mode='ave')(BDLSTM_output)
   # BDLSTM_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(BDLSTM_output)
    BDLSTM_output = Attention(10)(BDLSTM_output)

    NAME = "2-AttBDLSTMS___Wo_H_W-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[BDLSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def LSTMS(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True)(speed_input)
   # main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
   # main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
   # LSTM_output =LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=) (main_output)
    LSTM_output = Attention(6)(main_output)

    NAME = "4__LSTMS_-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def SBU_LSTMS(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')

    lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(lstm_output)
    lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(lstm_output)
    lstm_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(lstm_output)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2])(lstm_output)

    NAME = "4___SBULSTM_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)

    final_model = Model(input=[speed_input], output=[main_output])

    final_model.summary()

    final_model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def LSTMS_DNN(X, Y, epochs=30, validation_split=0.2, patience=10):
    model = Sequential()
    model.add(LSTM(output_dim=X.shape[2], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(output_dim=X.shape[2], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(output_dim=X.shape[2], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(output_dim=X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(X.shape[2]))
    NAME = "4__LSTM-DNN_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def AttLSTMS(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True)(speed_input)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    main_output =LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True) (main_output)
    LSTM_output = Attention(10)(main_output)

    NAME = "4__AttLSTMS_-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def BDLSTM(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]),  output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(main_output)
    BDLSTM_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True),merge_mode='ave')(BDLSTM_output)
    BDLSTM_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=False), merge_mode='ave')(BDLSTM_output)
   # BDLSTM_output = GRU(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2])(main_output)

    NAME = "4-BDLSTMS____-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[BDLSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def AttLSTMS_DNN(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True)(speed_input)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    main_output =LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True) (main_output)
    LSTM_output = Attention(10)(main_output)
    LSTM_output=(Dense(X.shape[2]))(LSTM_output)
    NAME = "3__AttLSTMS_DNN-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def BDLSTM_DNN(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), activation='sigmoid',  output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    main_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=False),merge_mode='ave')(main_output)
    BDLSTM_output = (Dense(X.shape[2]))(main_output)

    NAME = "2-BDLSTMS_DNN____-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[BDLSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def LSTMS_GRU(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True)(speed_input)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    main_output =LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True) (main_output)
    LSTM_output = GRU(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2])(main_output)

    NAME = "4__LSTMS_GRU_-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def LSTMS_ATTGRU(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = LSTM(input_shape = (X.shape[1], X.shape[2]), output_dim = X.shape[2], return_sequences=True)(speed_input)
    main_output = LSTM(input_shape=(X.shape[1], X.shape[2]), output_dim=X.shape[2], return_sequences=True)(main_output)
    LSTM_output = GRU(input_shape=(X.shape[1], X.shape[2]), return_sequences='True',output_dim=X.shape[2])(main_output)
    LSTM_output = Attention(10)(LSTM_output)

    NAME = "2___LSTMS__ATT-GRU_-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def BDLSTMS_ATTGRU(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional( LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True),merge_mode='ave')(speed_input)
    main_output = Bidirectional(LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True),merge_mode='ave')(main_output)
    LSTM_output = GRU(input_shape=(X.shape[1], X.shape[2]), return_sequences='False', output_dim=X.shape[2])(main_output)

    LSTM_output = Attention(10)(LSTM_output)

    NAME = "1___BDLSTMS__-GRU_-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def GRU(X, Y, epochs=30, validation_split=0.2, patience=10):
    model = Sequential()

    model.add(keras.layers.GRU(X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    #model.add(Dense(X.shape[2]))
    NAME = "1____GRU_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history
def deepAutoencoder(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')

    encoded = Dense(128, activation='relu')(speed_input)
    encoded = Dense(64, activation='relu')(encoded)
    #  sparsity concept  activity_regularizer=regularizers.l1(10e-5)// KL divergence
   # encoded = Dense(32, activation='sigmoid')(encoded)

   # decoded = Dense(64, activation='relu')(encoded)
   # decoded = Dense(128, activation='relu')(decoded)
   # decoded = Dense(32, activation='sigmoid')(decoded)

    NAME = "deepAutoencoder".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[encoded])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def FCN(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)

    y = Conv1D(128, 1, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)

    final_output = Dense(323,activation='sigmoid')(FCN_output)

    NAME = "2_FCN--rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def CNN(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    model = Sequential()
    model.add(Conv1D(128,kernel_size = 1, activation='relu', input_shape=(X.shape[1], X.shape[2])))
   # model.add(MaxPooling1D(pool_size=(2)))
    model.add(GlobalMaxPooling1D())

    model.add(Activation('sigmoid'))
    model.add(Dense(323))

    NAME = "1____CNN_-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True)
    model.summary()
    model.compile(loss='mse', optimizer='rmsprop')

    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    model.fit(X, Y, validation_split=0.2, nb_epoch=epochs, callbacks=[history, earlyStopping, tensorboard])

    return model, history

def CNN_BDLSTM(X, Y, epochs=30, validation_split=0.2, patience=10):

    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    y = Conv1D(128, 1, padding='same', kernel_initializer='he_uniform')(speed_input)
    y = GlobalMaxPooling1D()(y)
    FCN_output= Activation('sigmoid')(y)


    main_output = Bidirectional( LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional( LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(main_output)
    BDLSTM_output = Attention(10)(BDLSTM_output)

    combine_output = concatenate([BDLSTM_output, FCN_output])
    final_output = Dense(323, activation='sigmoid')(combine_output)
    NAME = "10_TL_____CNN_AttBDLSTM__-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history
def FCN_BDLSTM(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    FCN_output = GlobalAveragePooling1D()(y)
    main_output = Bidirectional( LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional( LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True), merge_mode='ave')(main_output)
    BDLSTM_output = Attention(6)(BDLSTM_output)

    combine_output = concatenate([BDLSTM_output, FCN_output])
    final_output = Dense(323, activation='sigmoid')(combine_output)
    NAME = "6_TL_____FCN_AttBDLSTM__-mse-rmse{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False,
                              write_grads=True)
    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop')
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history

def BDLSTMS_GRU(X, Y, epochs=30, validation_split=0.2, patience=10):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional( LSTM(input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', output_dim=X.shape[2], return_sequences=True),merge_mode='ave')(speed_input)
    LSTM_output  = keras.layers.GRU(X.shape[2], return_sequences=False, input_shape=(X.shape[1], X.shape[2]))(main_output)
   # LSTM_output = GRU(input_shape=(X.shape[1], X.shape[2]), return_sequences='False', output_dim=X.shape[2])(main_output)

   # LSTM_output = Attention(10)(LSTM_output)

    NAME = "1__BDLSTMS_GRU_updatded-mse-rmse{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )

    final_model = Model(input=[speed_input], output=[LSTM_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history


def AttBDLSTMS_FCN(X, Y, epochs=30, validation_split=0.2, patience=20):
    speed_input = Input(shape=(X.shape[1], X.shape[2]), name='speed')
    main_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]), activation='sigmoid', output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(speed_input)
    BDLSTM_output = Bidirectional(LSTM(input_shape = (X.shape[1], X.shape[2]),activation='sigmoid', output_dim = X.shape[2], return_sequences=True), merge_mode='ave')(main_output)
    BDLSTM_output = Attention(3)(BDLSTM_output)

    y = Permute((2, 1))(speed_input)
    y = Conv1D(128, 1, activation='sigmoid', padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)
    FCN_output = GlobalAveragePooling1D()(y)

    combine_output = concatenate([FCN_output, BDLSTM_output])

    final_output = Dense(323,activation='sigmoid')(combine_output)
    NAME = "TL-3_2-AttBDLSTMS_FCN__20p-{}".format(int(time.time()))
    tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=150, batch_size=32, write_graph=False, write_grads=True )
    final_model = Model(input=[speed_input], output=[final_output])
    final_model.summary()
    final_model.compile(loss='mse', optimizer='rmsprop' )
    history = LossHistory()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=patience, verbose=0, mode='auto')
    final_model.fit([X], Y, validation_split=0.2, epochs=epochs, callbacks=[history, earlyStopping, tensorboard])

    return final_model, history




