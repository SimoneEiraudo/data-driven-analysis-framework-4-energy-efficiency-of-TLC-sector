from termcolor import colored
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
import os
sys.path.insert(0, '../')  # add previous directory to path to load constants module
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pipeline.regression_fx import create_model
from pipeline.definitions import *

def train_ES_models(dataset, expl_var, display_loss=False):
    # ## Models training
    #create the list of IDs for the case studies and proper matrix to store performance results
    IDs=dataset.ID.unique()
    # select input and output variables
    dep_var = "P"
    ind_var = var_for_ES

    for caseStudy in IDs:
        df=dataset.loc[dataset.ID==caseStudy]
        x=pd.DataFrame(df[ind_var])
        y=pd.DataFrame(df[dep_var])

        #split test and train
        # split test and train
        if method_short== "ts":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        elif method_short== "ew":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        #create model
        fitted_uniES, _ = create_model(n_elbows=2)
        epochs = 50

        #define hyperparams
        batch = int(np.round(np.shape(x_train)[0] / 3, 1))
        adam = tf.keras.optimizers.Adam(learning_rate=0.1)
        lossFX = tf.keras.losses.MeanSquaredError()

        #compile and train
        fitted_uniES.compile(loss=lossFX, optimizer=adam, run_eagerly=True)
        history = fitted_uniES.fit(x_train, y_train,  validation_data=(x_test, y_test), epochs=epochs, batch_size=batch, shuffle=False, verbose=0)

        #calculate outputs
        y_pred_train=fitted_uniES.predict(x_train)
        y_pred_test=fitted_uniES.predict(x_test)

        #have a quick check to models performance ON THE SCALED DATASET (complete analysis of the models
        # performance is performed in notebook spr_01_03_00_models_train_check
        r2_train=r2_score(y_train, y_pred_train)
        r2_test=r2_score(y_test, y_pred_test)
        mae_train= mean_absolute_error(y_train, y_pred_train)
        mae_test= mean_absolute_error(y_test, y_pred_test)
        print("\nModel performance for the scaled data from casestudy "+caseStudy+" is \n R2 (train/test): "+str(round(r2_train, 2))+"   "+str(round(r2_test, 2))+"\n MAE(train/test): "+str(round(mae_train, 4))+"   "+str(round(mae_test, 4)))

        train_warning= r2_test<0.6
        if (display_loss==True) | (train_warning==True):
            #display loss evolution over epochs
            plt.figure(figsize=(3.2,2.4))
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss [Cp]')
            plt.legend()
            plt.grid(True)
            #plt.show()

        #save model
        filename = "PINN_uniES_"+caseStudy
        fitted_uniES.save(PAR_MODELS+"/"+filename+".h5")
        print(colored("A trained model was saved for case study "+caseStudy+"\n\n", 'green'))
    return None

if __name__ == "__main__":
    # select the method employed to normalize data and split train/test
    method_short = "ts"  # that is, time serie, or ew, that is elementWise
    # ## Data import
    dataset = pd.read_csv(os.path.join(DATA_NORMALIZED, 'norm_' + method_short + '_buildings_dataset.csv'))

    # Check dataset
    if dataset.isna().sum().sum() != 0:
        print(colored("Warning: some nan values existed in the dataset. These have been removed.", "yellow"))
        dataset = dataset.dropna()

    # select the input variable for ES
    var_for_ES = "T_a"  # or "T_eq_12" "T_eq_24"
    # decide wheter to produce an image to check train and test loss over epochs
    display_loss = False

    train_ES_models(dataset, var_for_ES)