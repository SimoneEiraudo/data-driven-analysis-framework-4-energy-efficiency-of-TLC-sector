from termcolor import colored
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')  # add previous directory to path to load constants module
from pipeline.definitions import *
from regression_fx import set_hypertuning, tune_hyperparameters, long_training, history_loss, optimal_training


def train_blackbox_models(dataset):
    # ## Models training
    #create the list of IDs for the case studies and proper matrix to store performance results
    IDs=dataset.ID.unique()
    #set the number of desired model to be saved for each case study (these will be the best ones from the hyperparameters search procedure)
    models_per_case=1
    columns_names=[ "model_"+str(x) for x in range(10) ]
    compTime=pd.DataFrame(columns=columns_names)
    display_loss=False

    #select input and output variables
    ind_var=["T_a", "RH", "Wv", "atmP", "G", "s_Wa", "c_Wa", "s_H", "c_H", "s_D", "c_D", "dayType"]
    n_inputs=len(ind_var)
    dep_var="P"

    for caseStudy in IDs:
        #define modeled input
        df=dataset.loc[dataset.ID==caseStudy]
        x=pd.DataFrame(df[ind_var])
        y=pd.DataFrame(df[dep_var])

        #split test and train
        if method_short== "ts":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        elif method_short== "ew":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        #train model
        tuner=set_hypertuning(caseStudy, ind_var, model_type="blackbox")                     #set grid search procedure
        [stop_early, best_hps_list]=tune_hyperparameters(tuner, x_train, y_train, n_models=models_per_case)            #find best n_models combinations of hyperparameters
        for hps, col in zip(best_hps_list, compTime.columns):   #iterate over hyperparameters configurations
            start_time = datetime.now()
            [nonPar_model, history]=long_training(tuner, hps, x_train, y_train)       #train over 150 epochs
            best_epoch=history_loss(history)                                                     #get epoch corresponding to minimum loss function
            optimal_model=optimal_training(tuner, hps, best_epoch, x_train, y_train)  #retrain according to optimal epoch
            end_time=datetime.now()

            # calculate outputs
            y_pred_train = optimal_model.predict(x_train)
            y_pred_test = optimal_model.predict(x_test)

            #quick model performance check
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            print("\nModel performance from "+col+" for the scaled data from casestudy " + caseStudy + " is \n R2 (train/test): " + str(
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

            #store computational time and save models
            compTime.loc[caseStudy, col] = (end_time - start_time).total_seconds()

            #save model
            optimal_model.save(BLACKBOX_MODELS+'/blackbox_' + col + '_' + caseStudy + '.h5')
    return compTime

if __name__ == "__main__":

    # select the method employed to normalize data and split train/test
    method_short = "ts"  # that is, time serie, or ew, that is elementWise

    # Import a case study dataset using pandas
    dataset = pd.read_csv(os.path.join(DATA_NORMALIZED, 'norm_' + method_short + '_buildings_dataset.csv'))

    # Check dataset
    if dataset.isna().sum().sum() != 0:
        print(colored("Warning: some nan values existed in the dataset. These have been removed.", "yellow"))
        dataset = dataset.dropna()

    compTime=train_blackbox_models(dataset)

    file_name = "blackbox_models_compTime"

    with open(RESULTS + "/" + file_name + ".pkl", 'wb') as f:
        pickle.dump(compTime, f)



