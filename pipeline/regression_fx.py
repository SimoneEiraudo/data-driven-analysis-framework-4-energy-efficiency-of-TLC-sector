import pandas as pd
from keras import layers
from keras.models import *
from keras.layers import *
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import os
import sys
sys.path.insert(0, '../')  # add previous directory to path to load constants module

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


class realElbows(tf.keras.constraints.Constraint):
    #constraints weights to be non negative and to sum up to 1
  def __init__(self, ref_value=0):
    self.ref_value = ref_value

  def __call__(self, w):
    #impose weights to be non negative (else are set to zero)
    w = w  * tf.cast(tf.greater_equal(w, 0), tf.float32)
    tf.compat.v1.enable_eager_execution()
    #set a minimum weight to connetions to avoid dead of training
    w=w+0.005-0.005*tf.cast(tf.greater_equal(w, 0.005), tf.float32)
    return w

  def get_config(self):
    return {'ref_value': self.ref_value}

def set_hypertuning(ID, ind_var, model_type):
    # Instantiate the tuner and perform hypertuning
    hypertuning_path = './Hypertuning/' + model_type + "/" + ID
    if not os.path.exists(hypertuning_path):
      os.makedirs(hypertuning_path)
    n_inputs = len(ind_var)
    tuner = kt.Hyperband(model_builder,
                         # objective='val_accuracy',
                         objective='val_loss',
                         # max_epochs=10,
                         max_epochs=20,
                         factor=3,
                         # factor=2,
                         directory=hypertuning_path,
                         project_name='Pres_hybrid')
    return tuner


def model_builder(hp):
  n_inputs = 12  # fix this!!!
  input = tf.keras.Input(shape=(n_inputs,))
  model = keras.Sequential()

  # Tune the number of units in the Dense layers
  # Choose an optimal value between 8-128
  hp_units = hp.Int('units', min_value=8, max_value=256, step=2, sampling='log')

  # Tune the number of layers between 1-4
  # hp_layers = hp.Int('layers', min_value=1, max_value=4, step=1)
  # hp_layers = hp.Int('layers', min_value=1, max_value=4, step=1, sampling='linear')
  hp_layers = hp.Int('layers', min_value=1, max_value=4, step=2, sampling='log')

  # Tune activation function between relu, tanh and sigmoid
  # hp_activation = hp.Choice('activation', values=['relu','tanh','sigmoid'])
  hp_activation = hp.Choice('activation', values=['relu', 'tanh'])

  # Define architecture
  model.add(input)
  for i in range(hp_layers):
    # model.add(keras.layers.Dense(units=hp_units, activation='tanh'))
    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))
  # final linear 1-unit layer
  # model.add(keras.layers.Dense(1,activation=hp_activation))
  model.add(keras.layers.Dense(1))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3])

  # Tune the batch size
  # hp_batch = hp.Int('batch', min_value=32, max_value=256, step=2, sampling='log')

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                # metrics=['accuracy'])
                loss='mean_absolute_error')

  return model

def tune_hyperparameters(tuner, x_NN, y_NN, n_models=1):
  # Create a callback to stop training early after reaching a certain value for the validation loss.
  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

  # if batch_size is not specified, 32 is the default value
  batch_size = 128
  tuner.search(x_NN, y_NN, epochs=25, validation_split=0.2, batch_size=batch_size, callbacks=[stop_early])

  # Get the optimal hyperparameters
  best_hps_list = tuner.get_best_hyperparameters(num_trials=n_models)
  if len(best_hps_list) == 1:
    print(f"""
        The hyperparameter search is complete! The optimal number of layers, number of units, 
        learning rate and activation function are:
        n_layers: {best_hps_list[0].get('layers')}
        n_units: {best_hps_list[0].get('units')}
        learning_rate: {best_hps_list[0].get('learning_rate')}
        activation_function: {best_hps_list[0].get('activation')}
        """)
  return stop_early, best_hps_list

def long_training(tuner, best_hps, x_NN, y_NN_unmodeled):
  # Train the model with optimal hyperparameters
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                   patience=10, min_lr=1e-6)
  # Create a callback to stop training early after reaching a constant value for the validation loss.
  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

  # Build the model with the optimal hyperparameters and train it on the data for 500 epochs
  batch_size = 128
  n_inputs = np.shape(x_NN)[1]
  nonPar_model = tuner.hypermodel.build(best_hps)
  history = nonPar_model.fit(x_NN, y_NN_unmodeled, epochs=150, batch_size=batch_size,
                             validation_split=0.2, callbacks=[reduce_lr, stop_early])
  return nonPar_model, history

def history_loss(history, show_plot=False):
  if show_plot == True:
    # Plot the loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [P_res]')
    plt.legend()
    plt.grid(True)

  val_loss_per_epoch = history.history['val_loss']
  train_loss_per_epoch = history.history['loss']
  best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
  print('Best epoch: %d' % (best_epoch,))
  return best_epoch

def optimal_training(tuner, best_hps, best_epoch, x_NN, y_NN_unmodeled):
  # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
  hypermodel = tuner.hypermodel.build(best_hps)
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                   patience=10, min_lr=1e-6)
  # Retrain the model
  hypermodel.fit(x_NN, y_NN_unmodeled, epochs=best_epoch, batch_size=128, validation_split=0.2, callbacks=[reduce_lr])
  return hypermodel

def create_model(n_elbows=1, approach='Daily', expect_CLC_activation_step='No'):
  if (approach == 'Daily') | (approach == 'Hourly'):
    input_layer = Input(shape=(1,))
    eq_T = input_layer
  elif approach == 'Multi_hour':
    input_layer = Input(shape=(24,))
    # create a layer to calculate a weighted sum of hourly temperatures recorder in the past 24 hours
    kernel_eq_T_Initializer = tf.keras.initializers.RandomNormal(mean=1 / 24, stddev=0.00001)
    eq_T = Dense(1, activation='linear', kernel_initializer=kernel_eq_T_Initializer, kernel_constraint=WeightedSum(),
                 use_bias=False)(input_layer)
  else:
    print("WARNING: Available approaches are 'Daily', 'Hourly' or 'Multi_hour'.")
  if expect_CLC_activation_step == 'No':
    # define kernel and bias for linear branch (relu)
    kerInit_relu = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01, seed=42)
    biasInit_relu = tf.keras.initializers.RandomNormal(mean=-0.1, stddev=0.1, seed=42)
    # define kernel and bias for linear branch (relu)/ set them to extremely low values to have no impact on the network
    kerInit_step = tf.keras.initializers.RandomNormal(mean=-100, stddev=0.01, seed=42)
    biasInit_step = tf.keras.initializers.RandomNormal(mean=-100, stddev=0.00001, seed=42)

    # create linear branch
    linear_ES_branch = Dense(n_elbows, activation='relu', kernel_initializer=kerInit_relu,
                             bias_initializer=biasInit_relu, kernel_constraint=realElbows())(eq_T)
    # create step branch/ set them to untrainable to freeze the step branch
    step_ES_branch = Dense(1, activation='sigmoid', kernel_initializer=kerInit_step,
                           bias_initializer=biasInit_step, trainable=False)(eq_T)
    step_ES_branch_B = Dense(1, activation='sigmoid', kernel_initializer=kerInit_step,
                             bias_initializer=biasInit_step, trainable=False)(step_ES_branch)

  elif expect_CLC_activation_step == 'Yes':
    # define kernel and bias for linear branch (relu)
    kerInit_relu = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)
    biasInit_relu = tf.keras.initializers.RandomNormal(mean=-0.1, stddev=0.00001)
    # define kernel and bias for linear branch (relu)/
    kerInit_step = tf.keras.initializers.RandomNormal(mean=15, stddev=0.01)
    biasInit_step = tf.keras.initializers.RandomNormal(mean=-8, stddev=0.00001)

    # create linear branch
    linear_ES_branch = Dense(n_elbows, activation='relu', kernel_initializer=kerInit_relu,
                             bias_initializer=biasInit_relu, kernel_constraint=tf.keras.constraints.NonNeg())(eq_T)
    # create step branch/ set them to untrainable to freeze the step branch
    step_ES_branch = Dense(1, activation='sigmoid', kernel_initializer=kerInit_step, bias_initializer=biasInit_step)(
      eq_T)
    step_ES_branch_B = Dense(1, activation='sigmoid', kernel_initializer=kerInit_step,
                             bias_initializer=biasInit_step)(step_ES_branch)

  # define kernel and bias f
  # or aggregation layer
  kerInit_aggregate = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.05)
  biasInit_aggregate = 'zeros'
  aggregate = Dense(1, kernel_initializer=kerInit_aggregate, bias_initializer=biasInit_aggregate,
                    kernel_constraint=tf.keras.constraints.NonNeg())(
    tf.concat([linear_ES_branch, step_ES_branch_B], axis=1))

  model = Model(inputs=input_layer, outputs=aggregate)
  extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
  return model, extractor