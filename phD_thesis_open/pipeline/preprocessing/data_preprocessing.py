import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import statistics
from pipeline.definitions import *
from os import listdir
from os.path import isfile, join
from pipeline.preprocessing_fx import GRbasedOutliersDet, calculateGradient, statistical_prepro
import csv



def preprocess_data(raw_data):
    data = raw_data.copy()

    # Remove columns with many NaN values
    data = data.drop(['UV'], axis=1)

    ### Preprocessing method 1
    # perform gradient based data-filtering to attempt just removing measurement errors
    IDs = data.ID.unique()
    clean_data = data.copy()
    load_ts = pd.DataFrame()

    for caseStudy in IDs:
        df_caseStudy = clean_data.loc[clean_data.ID == caseStudy, :]
        load_ts[caseStudy] = df_caseStudy.P.values

        IR = calculateGradient(load_ts)
        Emin = load_ts.nsmallest(10, columns=load_ts).mean() * 0.85
        [filteredTs, anomalyTypesDF, abnormalPointsDF] = GRbasedOutliersDet(load_ts, IR, Emin)

    #drop unreliable buildings
    reliable_IDs=filteredTs.columns
    clean_data=clean_data.loc[clean_data.ID.isin(reliable_IDs), :]

    for caseStudy in reliable_IDs:
        clean_data.loc[clean_data.ID == caseStudy, "P"] = filteredTs[caseStudy].values

    ### Preprocessing method 2
    # perform temperature curve-based data-filtering for more precise data-filtering

    # Define output(s)
    dep_var = ['P']
    # include all the other variables in the x datasets (inputs), except the case study key (ID)
    ind_var = [var for var in list(data.columns) if ((var not in dep_var) and (var not in ["ID"]))]

    # # create dataframes
    # X_data = data[ind_var]
    # Y_data = data[dep_var]
    # name_data = data['ID']
    #
    # # Convert dataframes to numpy arrays
    # X_data = X_data.to_numpy(dtype='float64')
    # Y_data = Y_data.to_numpy(dtype='float64')

    # Discard anomalous data from power curve (case study by caser Study)
    data = data.dropna(axis=0)
    good_data = data.copy()

    for caseStudy in IDs:
        # access a case study
        df_caseStudy = good_data.loc[good_data.ID == caseStudy, :]
        good_data = good_data.loc[good_data.ID != caseStudy, :]  # this is like a drop
        # preprocess the case study
        good_df_caseStudy, P_estimated, T_estimate = statistical_prepro(df_caseStudy)
        # store the preprocessed results
        good_data = pd.concat([good_data, good_df_caseStudy])

    raw_points=np.shape(raw_data)[0]
    points=np.shape(data)[0]
    goodPoints=np.shape(good_data)[0]
    cleanPoints = np.shape(clean_data)[0]
    buildings = len(raw_data.ID.unique())
    clean_buildings=len(clean_data.ID.unique())
    print("The gradient-based preprocessing detected " + str(buildings-clean_buildings) + " unreliable buildings")
    print("The gradient-based preprocessing step reduced the dataset from "+str(raw_points)+" to "+str(cleanPoints))
    #print("In particular\n - "+str(raw_points-points)+" measures were deleted because they contained nan values \n - "+str(points-cleanPoints)+" measures were deleted for being outliers")
    print("The statistical preprocessing step reduced the dataset from "+str(raw_points)+" to "+str(goodPoints))
    print("In particular\n - "+str(raw_points-points)+" measures were deleted because they contained nan values \n - "+str(points-goodPoints)+" measures were deleted for being outliers")

    return good_data, clean_data

if __name__ == "__main__":
    # Get dataset and load it as a Pandas DataFrame
    raw_data = pd.read_csv(os.path.join(DATA_RAW, 'all_buildings_dataset'))

    good_data, clean_data=preprocess_data(raw_data)

    # # # Save filtered Dataset
    with open(os.path.join(DATA_FILTERED, 'buildings_dataset_filtered.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(list(good_data.columns))
        # write multiple rows
        writer.writerows(good_data.values)

    with open(os.path.join(DATA_CLEAN, 'buildings_dataset_clean.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(list(clean_data.columns))
        # write multiple rows
        writer.writerows(clean_data.values)

    print("\nThe reduced datasets were produced and saved.")