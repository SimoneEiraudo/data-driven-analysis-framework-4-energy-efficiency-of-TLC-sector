
import pandas as pd
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from termcolor import colored
from sklearn.model_selection import train_test_split
import pickle

import sys
sys.path.insert(0, '../')  # add previous directory to path to load constants module

from pipeline.regression_fx import realElbows
from pipeline.definitions import *
from pipeline.monitoring_fx import cusum_tab, plot_cusum_tab

def calculate_residuals(dataset, scalers_dict):


    #create two dictionaries to load models
    ES_models={}
    semiPar_models={}

    ES_models_files=[f for f in listdir(PAR_MODELS) if isfile(join(PAR_MODELS, f))]
    semiPar_models_files=[f for f in listdir(SEMIPAR_MODELS) if isfile(join(SEMIPAR_MODELS, f))]

    #load models
    IDs=dataset.ID.unique()
    for caseStudy in IDs:
        #get the model(s) for the case study
        caseStudy_semiPar_models=[f for f in semiPar_models_files if caseStudy in f]
        caseStudy_ES_models = [f for f in ES_models_files if caseStudy in f]
        #check existance of 1 and no more models for the cast study
        if (len(caseStudy_semiPar_models)==0) | (len(caseStudy_ES_models)==0):
            print(colored("ERROR: No trained model has been found for case study "+caseStudy, "red"))
        elif len(caseStudy_ES_models)+len(caseStudy_semiPar_models)>2:
            print("Multiple models have been found for case study "+caseStudy+"\nThe following models are available:\n")
            print(caseStudy_ES_models)
            print(caseStudy_semiPar_models)
            modelname=input("Select one of the models ")
        else:
            ES_modelname=caseStudy_ES_models[0]
            semiPar_modelname = caseStudy_semiPar_models[0]

        #load models
        ES_model = tf.keras.models.load_model(PAR_MODELS + '/' + ES_modelname,custom_objects={'realElbows': realElbows})
        ES_models[caseStudy] = ES_model
        semiPar_model=tf.keras.models.load_model(SEMIPAR_MODELS+'/'+semiPar_modelname, custom_objects={'realElbows':realElbows})
        semiPar_models[caseStudy]=semiPar_model

    #create a dictionary to store residuals (keys : building, model)
    predictions={}
    residuals={}

    #select input and output variables
    ind_var=["T_a", "RH", "Wv", "atmP", "G", "s_Wa", "c_Wa", "s_H", "c_H", "s_D", "c_D", "dayType"]
    n_inputs=len(ind_var)
    modeled_ind_var="T_a"
    dep_var="P"

    for caseStudy in IDs:
        # define modeled input and select regression models
        df = dataset.loc[dataset.ID == caseStudy]
        ES_model = ES_models[caseStudy]
        semiPar_model = semiPar_models[caseStudy]

        x = pd.DataFrame(df[ind_var])
        x_modeled = pd.DataFrame(df[modeled_ind_var])
        y = pd.DataFrame(df[dep_var])
        # split test and train
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)

        x_modeled_train = pd.DataFrame(x_train[modeled_ind_var])
        x_modeled_test = pd.DataFrame(x_test[modeled_ind_var])

        # pd.options.mode.chained_assignment = None  # default='warn', use it to hide the SettingCopyWarning

        # calculate parametric model output and residuals
        y_NN_ES_pred = ES_model.predict(x_modeled, verbose=0)
        y_NN_semiPar_pred = semiPar_model.predict((x_modeled, x), verbose=0)


        # rescale real outputs
        y_real = scalers_dict[caseStudy, "y"].inverse_transform(y)
        y_real_train = scalers_dict[caseStudy, "y"].inverse_transform(y_train)
        y_real_test = scalers_dict[caseStudy, "y"].inverse_transform(y_test)

        for approach, y_NN_pred in zip(["ES", "semiPar"], [y_NN_ES_pred, y_NN_semiPar_pred]):
            # split test and train for predicted values
            _, _, y_NN_pred_train, y_NN_pred_test = train_test_split(x, y_NN_pred, test_size=0.33, shuffle=False)

            # predictions of the full set
            y_pred = scalers_dict[caseStudy, "y"].inverse_transform(y_NN_pred)
            predictions[approach, caseStudy] = pd.Series(y_pred.flatten())
            res=pd.Series(y_real.flatten()-y_pred.flatten())
            res[res.isna()] = 0 #do this to ensure continuity in all time series and hence be able to later compute cusum
            residuals[approach, caseStudy] = res

            # # store non-rescaled predictions
            # NN_predictions[approach, caseStudy, model_n, "Train"] = y_NN_pred_train
            # NN_predictions[approach, caseStudy, model_n, "Test"] = y_NN_pred_test
            # NN_predictions[approach, caseStudy, model_n, "Whole Set"] = y_NN_pred

        print(colored("Predictions and residuals from building " +caseStudy+ " correctly retrieved", "green"))
    return residuals

if __name__ == "__main__":
    # load dataset
    # select the method employed to normalize data and split train/test
    method_short = "ts"  # that is, time serie, or ew, that is elementWise

    dataset = pd.read_csv(os.path.join(DATA_NORMALIZED, 'norm_' + method_short + '_buildings_dataset.csv'))
    print("Shape of dataset: " + str(dataset.shape))
    dataset.tail()

    # load scalers
    file_name = "scalers_" + method_short
    with open(DATA + "/" + file_name + ".pkl", 'rb') as f:
        scalers_dict = pickle.load(f)

    residuals=calculate_residuals(dataset, scalers_dict)

    #save scalers (rescaled data will be needed for the post-processing step
    file_name="residuals"

    with open(RESULTS+"/"+file_name+".pkl", 'wb') as f:
        pickle.dump(residuals, f)
