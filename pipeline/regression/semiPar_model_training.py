from termcolor import colored
import pandas as pd
import pickle
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')  # add previous directory to path to load constants module
from pipeline.regression_fx import realElbows, set_hypertuning, tune_hyperparameters, long_training, history_loss, optimal_training
from pipeline.definitions import *


def train_semiPar_models(dataset):

    # ## Models training
    #create the list of IDs for the case studies and proper matrix to store performance results
    IDs=dataset.ID.unique()

    #Load parametric regression sub-models into a dictionary
    models={}
    #get all available parametric models
    ES_models = [f for f in listdir(PAR_MODELS) if isfile(join(PAR_MODELS, f))]
    for caseStudy in IDs:
        #get the model(s) for the case study
        caseStudy_models=[f for f in ES_models if caseStudy in f]
        #check existance of 1 and no more models for the cast study
        if len(caseStudy_models)==0:
            print(colored("ERROR: No trained model has been found for case study "+caseStudy, "red"))
        elif len(caseStudy_models)>1:
            print("Multiple models have been found for case study "+caseStudy+"\nThe following models are available:\n")
            print(caseStudy_models)
            modelname=input("Select one of the models ")
        else:
            modelname=caseStudy_models[0]

        #load models
        ES_model=tf.keras.models.load_model(PAR_MODELS+'/'+modelname, custom_objects={'realElbows':realElbows})
        models[caseStudy]=ES_model
        #model.summary()

    #set the number of desired model to be saved for each case study (these will be the best ones from the hyperparameters search procedure)
    models_per_case=1
    columns_names=[ "model_"+str(x) for x in range(10) ]
    compTime=pd.DataFrame(columns=columns_names)
    display_loss=False

    #select input and output variables
    ind_var=["T_a", "RH", "Wv", "atmP", "G", "s_Wa", "c_Wa", "s_H", "c_H", "s_D", "c_D", "dayType"]
    n_inputs=len(ind_var)
    modeled_ind_var="T_a"
    dep_var="P"

    #create new columns in dataset
    dataset["P_phys"]=np.nan
    dataset["P_res"]=np.nan

    for caseStudy in IDs:
        #define modeled input
        df=dataset.loc[dataset.ID==caseStudy]
        x=pd.DataFrame(df[ind_var])
        x_modeled=pd.DataFrame(df[modeled_ind_var])
        y=pd.DataFrame(df[dep_var])

        # split test and train
        if method_short == "ts":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        elif method_short == "ew":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        x_modeled_train=pd.DataFrame(x_train[modeled_ind_var])
        x_modeled_test=pd.DataFrame(x_test[modeled_ind_var])

        #pd.options.mode.chained_assignment = None  # default='warn', use it to hide the SettingCopyWarning
        #load case study model
        ES_model=models[caseStudy]

        # calculate parametric model output and residuals
        y_ES_pred=ES_model.predict(x_modeled)
        y_res=y-y_ES_pred
        # calculate separately the residuals for the train set
        y_ES_pred_train=ES_model.predict(x_modeled_train)
        y_res_train=y_train-y_ES_pred_train

        #calculate and store ES predictions and residuals
        dataset.loc[dataset.ID==caseStudy, "P_phys"]=y_ES_pred
        dataset.loc[dataset.ID==caseStudy, "P_res"]=y_res

        # train model
        tuner = set_hypertuning(caseStudy, ind_var, model_type="hybrid")  # set grid search procedure
        [stop_early, best_hps_list] = tune_hyperparameters(tuner, x_train, y_res_train,
                                                           n_models=models_per_case)  # find best n_models combinations of hyperparameters
        for hps, col in zip(best_hps_list, compTime.columns):  # iterate over hyperparameters configurations
            start_time = datetime.now()
            [nonPar_model, history] = long_training(tuner, hps, x_train, y_res_train)  # train over 150 epochs
            best_epoch = history_loss(history)  # get epoch corresponding to minimum loss function
            optimal_model = optimal_training(tuner, hps, best_epoch, x_train, y_res_train)  # retrain according to optimal epoch
            end_time = datetime.now()

            # calculate predictions from the non parametric sub-model
            y_res_pred_train = optimal_model.predict(x_train)
            y_res_pred_test = optimal_model.predict(x_test)

            ## join models
            input1 = tf.keras.Input(shape=(np.shape(x_modeled)[1],))
            input2 = tf.keras.Input(shape=(np.shape(x)[1],))

            output1 = ES_model(input1)
            output2 = optimal_model(input2)

            # Global model
            global_output = tf.keras.layers.Add()([output1, output2])

            semiPar_model = tf.keras.Model(inputs=[input1, input2], outputs=global_output)
            #model.summary()
            #tf.keras.utils.plot_model(model, "../Models/Global_model.png", show_shapes=True)

            #calculate global model output
            y_global_pred = semiPar_model.predict((x_modeled, x))
            #sepatately calculate train and test output
            y_global_pred_train = semiPar_model.predict((x_modeled_train, x_train))
            y_global_pred_test = semiPar_model.predict((x_modeled_test, x_test))

            # quick model performance check (better do this on the gloabal model, that is, first join models
            r2_train = r2_score(y_train, y_global_pred_train)
            r2_test = r2_score(y_test, y_global_pred_test )
            mae_train = mean_absolute_error(y_train, y_global_pred_train)
            mae_test = mean_absolute_error(y_test, y_global_pred_test )

            print(
                "\nModel performance from " + col + " for the scaled data from casestudy " + caseStudy + " is \n R2 (train/test): " + str(
                    round(r2_train, 2)) + "   " + str(round(r2_test, 2)) + "\n MAE(train/test): " + str(
                    round(mae_train, 4)) + "   " + str(round(mae_test, 4)))

            train_warning = r2_test < 0.6
            if (display_loss == True) | (train_warning == True):
                # display loss evolution over epochs
                plt.figure(figsize=(3.2, 2.4))
                plt.plot(history.history['loss'], label='loss')
                plt.plot(history.history['val_loss'], label='val_loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss [Cp]')
                plt.legend()
                plt.grid(True)
                plt.show()

            # store computational time and save models
            compTime.loc[caseStudy, col] = (end_time - start_time).total_seconds()

            #save semiPar model
            semiPar_model.save(SEMIPAR_MODELS+'/semiPar_' + col + '_' + caseStudy + '.h5')
    return compTime

if __name__ == "__main__":
    # select the method employed to normalize data and split train/test
    method_short = "ts"  # that is, time serie, or ew, that is elementWise

    dataset = pd.read_csv(os.path.join(DATA_NORMALIZED, 'norm_' + method_short + '_buildings_dataset.csv'))

    # Check dataset
    if dataset.isna().sum().sum() != 0:
        print(colored("Warning: some nan values existed in the dataset. These have been removed.", "yellow"))
        dataset = dataset.dropna()

    compTime=train_semiPar_models(dataset)

    file_name = "semiPar_models_compTime"

    with open(RESULTS + "/" + file_name + ".pkl", 'wb') as f:
        pickle.dump(compTime, f)

