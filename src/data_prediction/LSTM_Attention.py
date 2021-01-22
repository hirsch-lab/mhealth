''' The objective is to predict the evolution of certain vital-parameters
 by evaluating and training the recurrent neural network (RNN)
 on continuous measured vital signs and additionally incorporating attention
 to improve the interpretability and to understand the decision making of
 the model on each data point per patient.
 Reference: The LSTM network was inspired by https://levelup.gitconnected.com/
 building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb '''

import os

import numpy as np
import tensorflow
import time
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector, dot, concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from patient.patient_data_loader import PatientDataLoader


#####################################################################################################

class AttentionLSTM:
    loader = PatientDataLoader()

    def import_data_imove(self, dir_name):
        network_input_left = []
        network_labels_left = []
        # network_input_right = []
        # network_labels_right = []

        for filename in os.listdir(dir_name):
            if filename.endswith('csv'):  # and fnmatch.fnmatch(filename, "*left"):
                df = self.loader.load_everion_patient_data(dir_name, filename, ',')
                network_input_left.append(df.HR)  # df.BloodPressure, df.Energy, df.RespRate)
                network_labels_left.append(df.Classification)

            # elif filename.endswith('*.csv') and fnmatch.fnmatch(filename, "*righ"):
            #     df = pd.DataFrame(pd.read_csv(filename, sep=';'))
            #     network_input_right = [df.HR, df.BloodPressure, df.Energy, df.RespRate, df.objtemp]
            #     network_labels_right = [df.Classification]
            #
            # else:
            #     current_path = "".join((dir_name, "/", filename))
            #     if os.path.isdir(current_path):
            #         self.import_data_imove(current_path)

        input_train, input_test, label_train, label_test = train_test_split(network_input_left,
                                                                            network_labels_left, test_size=0.20)
        input_train = np.array(input_train)
        input_test = np.array(input_test)
        label_train = np.array(label_train)
        label_test = np.array(label_test)

        input_train = input_train.reshape((input_train.shape[0], 1, input_train.shape[1]))
        input_test = input_test.reshape((input_test.shape[0], 1, input_test.shape[1]))

        self.define_model(input_train, input_test, label_train, label_test)

    #####################################################################################################

    def import_data_covid(self, dir_name):
        network_input = []
        network_labels = []

        for filename in os.listdir(dir_name):
            if filename.endswith('csv'):
                df = self.loader.load_everion_patient_data(dir_name, filename, ',')
                network_input.append(df.heart_rate)
                network_labels.append(df.core_temperature)

        input_train, input_test, label_train, label_test = train_test_split(network_input,
                                                                            network_labels, test_size=0.10)

        input_train = np.array(input_train)
        input_test = np.array(input_test)
        label_train = np.array(label_train)
        label_test = np.array(label_test)

        input_train = input_train.reshape((input_train.shape[0], 1, input_train.shape[1]))
        input_test = input_test.reshape((input_test.shape[0], 1, input_test.shape[1]))
        label_train = label_train.reshape((label_train.shape[0], 1, input_test.shape[2]))
        label_test = label_test.reshape((label_test.shape[0], 1, label_test.shape[1]))

        input_train_x = Input(shape=(input_train.shape[1], input_train.shape[2]))
        label_train_y = Input(shape=(label_train.shape[1], label_train.shape[2]))
        print(input_train.shape)
        print(label_train.shape)

        self.define_model(input_train, input_test, label_train, label_test, input_train_x, label_train_y)

        #####################################################################################################

    def import_data_ukbb(self, dir_name):
        tic = time.perf_counter()

        network_input = []
        network_labels = []

        for filename in os.listdir(dir_name):
            if filename.endswith('csv'):
                df = self.loader.load_everion_patient_data(dir_name, filename, ',')
                network_input.append(df.HR)
                network_labels.append(df.objtemp)

        # input_train, input_val, label_train, label_val = train_test_split(network_input,
        #                                                                     network_labels, test_size=0.10)

        input_train, input_val, label_train, label_val = train_test_split(network_input,
                                                                          network_labels, test_size=0.2, random_state=1)

        input_train, input_test, label_train, label_test = train_test_split(input_train,
                                                                            label_train, test_size=0.25, random_state=1)

        input_train = np.array(input_train)
        input_val = np.array(input_val)
        label_train = np.array(label_train)
        label_val = np.array(label_val)
        input_test = np.array(input_test)
        label_test = np.array(label_test)

        input_train = input_train.reshape((input_train.shape[0], 1, input_train.shape[1]))
        input_val = input_val.reshape((input_val.shape[0], 1, input_val.shape[1]))
        label_train = label_train.reshape((label_train.shape[0], 1, label_train.shape[1]))
        label_val = label_val.reshape((label_val.shape[0], 1, label_val.shape[1]))
        input_test = input_test.reshape((input_test.shape[0], 1, input_test.shape[1]))
        label_test = label_test.reshape((label_test.shape[0], 1, label_test.shape[1]))

        toc = time.perf_counter()
        print(f"Imported data in {toc - tic:0.4f} seconds")

        input_train_x = Input(shape=(input_train.shape[1], input_train.shape[2]))
        label_train_y = Input(shape=(label_train.shape[1], label_train.shape[2]))
        print(input_train.shape)
        print(label_train.shape)

        # toc = time.perf_counter()
        # print(f"Imported data in {toc - tic:0.4f} seconds")

        self.define_model(input_train, input_val, label_train, label_val, input_train_x, label_train_y, input_test,
                          label_test)

    #####################################################################################################

    ### ATTENTION-BASED LSTM MODEL ###

    def define_model(self, input_train, input_val, label_train, label_val, input_train_x, label_train_y, input_test,
                     label_test):

        # Set random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        n_hidden = 100  # number of hidden layers

        # Encoder LSTM
        # return last hidden state, last cell state and stacked hidden states for alignment score calculation
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(n_hidden, activation='elu', dropout=0.2,
                                                               recurrent_dropout=0.2, return_state=True,
                                                               return_sequences=True, dtype='float64')(input_train_x)

        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        # Decoder LSTM
        decoder_input = RepeatVector(label_train_y.shape[1])(encoder_last_h)

        decoder_stack_h = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
                               return_state=False, return_sequences=True)(
            decoder_input, initial_state=[encoder_last_h, encoder_last_c])

        # Attention layer
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_stack_h], axes=[2, 1])
        context = BatchNormalization(momentum=0.6)(context)

        decoder_combined_context = concatenate([context, decoder_stack_h])

        out = TimeDistributed(Dense(label_train_y.shape[2]))(decoder_combined_context)

        model = Model(inputs=input_train_x, outputs=out)
        model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])
        model.summary()

        self.train_model(model, input_train, input_val, label_train, label_val, input_test, label_test)

    #####################################################################################################

    def train_model(self, model, input_train,
                    input_val, label_train, label_val, input_test, label_test):
        tic = time.perf_counter()

        history = model.fit(input_train, label_train, epochs=100, batch_size=72,
                            validation_data=(input_val, label_val), verbose=2, shuffle=False)
        pyplot.plot(history.history['mean_absolute_percentage_error'], label='train')
        pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='val')
        pyplot.ylim((0, 100))
        pyplot.ylabel('mean absolute percentage error')
        pyplot.xlabel('epoch')
        pyplot.title('train vs. test accuracy')
        pyplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        pyplot.show()

        model.save('att_LSTM_model.h5')
        toc = time.perf_counter()
        print(f"Model trained in {toc - tic:0.4f} seconds")

        self.evaluate_model(input_test, label_test)

    #####################################################################################################

    def evaluate_model(self, input_test, label_test):
        tic = time.perf_counter()

        model = load_model('att_LSTM_model.h5')
        # Evaluate the model on the test data
        print('Evaluate on test data')
        scores_test = model.evaluate(input_test, label_test, batch_size=72, verbose=0)
        print('test loss, test mae:', scores_test)

        # Generate predictions on new data
        print("Generate predictions")
        prediction_input = model.predict(input_test)
        prediction_label = model.predict(label_test)
        print("predictions input and label:", prediction_input, prediction_label,
              prediction_input.shape, prediction_label.shape)

        toc = time.perf_counter()
        print(f"Model evaluated in {toc - tic:0.4f} seconds")

        self.save_model(model)

    #####################################################################################################

    def save_model(self, model):
        # Save the model to a json file

        model_json = model.to_json()
        with open("/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/"
                  "UKBB/LSTM/150_test_data_att/changed_input/att_lstm_ukbb_v100.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/"
                           "UKBB/LSTM/150_test_data_att/changed_input/att_lstm_ukbb_v100.h5")

        plot_model(model, to_file='/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/'
                                  'UKBB/LSTM/150_test_data_att/changed_input/att_model_plot100.png',
                   show_shapes=True, show_layer_names=True)

#####################################################################################################

# if __name__ == '__main__':
#     dir_name = '/Users/reys/Desktop/ACLS_Master/Track_Module_2/dataset2_imove/training/training_left_classification/'
#     dir_name_test = '../resources/data_prediction_imove/left'
#     prediction = AttentionLSTM()
#     prediction.import_data_imove(dir_name_test)

# if __name__ == '__main__':
#     dir_name = '/Users/reys/Desktop/ACLS_Master/Track_Module_2/dataset1_covid/training_copy_selected_4737/'
#     dir_name_test = '../resources/data_prediction_covid'
#     prediction = AttentionLSTM()
#     prediction.import_data_covid(dir_name)

if __name__ == '__main__':
    dir_name = '/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/UKBB/data_selected_34700/'
    prediction = AttentionLSTM()
    prediction.import_data_ukbb(dir_name)
