import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
import pickle
from pipeline.definitions import *

def normalization(dataset):
    #select input and output variables
    ind_var=["T_a", 'T_eq_12','T_eq_24', "RH", "Wv", "atmP", "G", "s_Wa", "c_Wa", "s_H", "c_H", "s_D", "c_D", "dayType"]
    n_inputs=len(ind_var)
    dep_var="P"
    caseStudy_key="ID"

    #create a dictionary to store and save the scalers
    scalers_dict = {}
    IDs=dataset.ID.unique()
    norm_data=dataset[[dep_var]+ind_var+[caseStudy_key]]

    for caseStudy in IDs:
        # define modeled input
        df = dataset.loc[dataset.ID == caseStudy]
        x = pd.DataFrame(df[ind_var])
        y = pd.DataFrame(df[dep_var])

        # split test and train
        if approach == "timeSerieSplit":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)
        elif approach == "elementShuffleSplit":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        scaler_x = MinMaxScaler()
        scaler_x.fit(x_train)
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_train)
        #store the scalers into the dictionary
        scalers_dict[caseStudy, "x"]=scaler_x
        scalers_dict[caseStudy, "y"] = scaler_y

        #apply scalers to the whole train+test set
        norm_ind_var=scaler_x.transform(x)
        norm_dep_var = scaler_y.transform(y)
        norm_data.loc[norm_data.ID==caseStudy, dep_var]=norm_dep_var
        norm_data.loc[norm_data.ID==caseStudy, ind_var] = norm_ind_var

    return norm_data, scalers_dict

if __name__ == "__main__":

    approach = "timeSerieSplit"  # or "elementShuffleSplit"
    if approach == "timeSerieSplit":
        dataset = pd.read_csv(os.path.join(DATA_CLEAN, 'buildings_dataset_clean.csv'))
        method_short = "ts"  # that is, time series
    elif approach == "elementShuffleSplit":
        dataset = pd.read_csv(os.path.join(DATA_FILTERED, 'buildings_dataset_filtered.csv'))
        method_short = "ew"  # that is, element wise

    norm_data, scalers_dict=normalization(dataset)
    #save scalers (rescaled data will be needed for the post-processing step
    file_name="scalers_"+method_short

    with open(DATA+"/"+file_name+".pkl", 'wb') as f:
        pickle.dump(scalers_dict, f)


    #save normalized dataframe

    with open(DATA_NORMALIZED+'/norm_'+method_short+'_buildings_dataset.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(list(norm_data.columns))
        # write multiple rows
        writer.writerows(norm_data.values)