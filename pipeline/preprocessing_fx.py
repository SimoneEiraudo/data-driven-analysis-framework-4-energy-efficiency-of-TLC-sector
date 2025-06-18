import logging
import math
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import statistics
from scipy.stats import median_abs_deviation

sys.path.append(str(Path(__file__).resolve().parents[2]))
logger = logging.getLogger()

def GRbasedOutliersDet(dfRaw, IR, Emin):
    df=dfRaw.copy()
    #calculate stats from the gradient time series and define error conditions
    sampIR = IR.mean()
    stdIR = IR.std()
    IRcum = IR.cumsum()
    # Define error as 3 times the standard deviation of gradient
    errorCond = 8 * stdIR

    #Create a DataFrame as a counter for outliers for each station
    outlCount=pd.DataFrame(index = df.columns, columns=['Total number of abnormal points'])
    anomCount=pd.DataFrame(index=df.columns, columns=['Puntual outliers', '2 Points Anomaly', 'Long-lasting Anomaly'])
    for pod in df.columns.tolist():
        # Create three counter according to length of anomalies as a counter for type of anomaly + 1 for count of total abnormal points
        [pointAnom, twoPointsAnom, longAnom, totalAnom]=4*[0]
        #find error according to confidence interval of the statistical distribution of gradient
        errors = (IR[pod] > sampIR.loc[pod] + errorCond.loc[pod]) | (IR[pod] < sampIR.loc[pod] - errorCond.loc[pod])
        while errors.any()==True:
            #define first outlier
            firstErr=errors.loc[errors==True].idxmin()
            #get last reliable using position firstErr-1 and get the corresponding gradient's cumulative
            lastRelIRcum=IRcum.loc[IRcum.index[IRcum.index.get_loc(firstErr)-1], pod]
            # define reliability condition limits according to 3*variance
            #upperReliabilityLimit=np.linspe
            hoursOfYear=len(df.index.unique())
            relCon = errorCond[pod] * np.linspace(1, hoursOfYear + 1 - IRcum.index.get_loc(firstErr),
                            hoursOfYear + 1 - IRcum.index.get_loc(firstErr)-1)#(range(1, hoursOfYear+1 - IRcum.index.get_loc(firstErr))) * errorCond[pod]
            #find next reliable value
            reliabilityArray = ((IRcum.loc[firstErr:, pod] - lastRelIRcum).abs() < relCon) & (df.loc[firstErr:, pod]>Emin.loc[pod])
            # check existance of a reliable value after the identified outlier
            if reliabilityArray.any() == True:
                #get the index of the first reliable value
                firstRel=reliabilityArray.idxmax()
                # calculate distance between error and first reliable value
                dist = firstRel - firstErr
                #print(firstRel)
                #print(firstErr)
                #print(dist)
            else:
                dist=df.index[-1]-firstErr

            #if length of anomaly is equal or longer than 2 days, drop the time serie it belongs to
            if dist>=48:#timedelta(days=2):
                totalAnom = hoursOfYear
                df=df.drop(pod, axis=1)
                break
            # else, if the anomaly is shorted than two days, update count of anomalies (type of anomaly defined by length)
            elif dist == 1:#timedelta(hours=1):
                pointAnom=pointAnom+1
            elif dist == 2:#timedelta(hours=2):
                twoPointsAnom=twoPointsAnom+1
            else:
                longAnom=longAnom+1
            #update count of total abnormal points
            totalAnom=totalAnom+dist
            #drop station if total number of abnormal points overcome 10% of the length of the time serie
            if totalAnom>=len(df)*0.1:
                totalAnom=hoursOfYear
                df = df.drop(pod, axis=1)

                break

            #drop value from abnormal measures (they will later be replace, see below, after end of the "while" loop)
            df.loc[firstErr:firstRel, pod]=np.nan
            errors[:firstRel]=False

        if (pod==df.columns).any():
            df.loc[:, pod] = df.loc[:, pod].interpolate()
        anomCount.loc[pod, :]=[pointAnom, twoPointsAnom, longAnom]

        # add number of outliers to counter
        outlCount.loc[pod, 'Total number of abnormal points'] = int(totalAnom)
    return df, anomCount, outlCount

def calculateGradient(df, nPlots=1):
#Calculate -gradient over time series
    #create a dataframe for the gradient of time series
    IR=pd.DataFrame(index=df.index, columns=df.columns)
    #first value of each single time serie should be nan
    IR.iloc[0]=np.nan
    #the rest of the values are calculated
    IR.iloc[1:, :]=np.diff(df, axis=0)
    return(IR)

def data_exploration(data, data_description, var1, var2=None, caseStudy="random", subplot=None, font_size=12):
    # select case study
    IDs = data.ID.unique()
    if caseStudy == "random":
        caseStudy = random.choice(IDs)
    elif caseStudy not in IDs:
        print("warning: the select case study does not exist!\nUse argument caseStudy to select a case study")
        return None
    df = data.loc[data["ID"] == caseStudy, :]

    # create a new graphical object if needed
    if subplot == None:
        fig, subplot = plt.subplots(1)
    # provide scatter plot if two variables were passed
    if var2 != None:
        subplot.scatter(df[var1], df[var2], s=3)
        y_label = data_description.loc[data_description["Variable_name"] == var2, "variable_label"].values[0]
        subplot.set_ylabel(y_label, fontsize=font_size)
    # provide histogram if one varible was passed
    else:
        subplot.hist(df[var1], bins=30)
        subplot.set_ylabel("Number of elements", fontsize=font_size)

    x_label = data_description.loc[data_description["Variable_name"] == var1, "variable_label"].values[0]
    subplot.set_xlabel(x_label, fontsize=font_size)
    subplot.tick_params(labelsize=font_size)
    return None

def check_outliers(data, good_data, data_description, P_estimated, T_estimated, caseStudy="random", subplot=None,
                   font_size=12):
    df_caseStudy = data.copy()
    df_good_caseStudy = good_data.copy()

    # create a new graphical object if needed
    if subplot == None:
        fig, subplot = plt.subplots(1)

    var1 = "T_a"
    var2 = "P"
    # get anomalous points
    df_anom = df_caseStudy.drop(df_good_caseStudy.index)
    subplot.scatter(df_good_caseStudy[var1], df_good_caseStudy[var2], s=3, color="blue", label='Normal points')
    subplot.scatter(df_anom[var1], df_anom[var2], s=4, facecolors='none', edgecolors='grey', label='Anomalous point')
    subplot.plot(T_estimated, P_estimated, 'r-', label='Power curve')
    y_label = data_description.loc[data_description["Variable_name"] == var2, "variable_label"].values[0]
    x_label = data_description.loc[data_description["Variable_name"] == var1, "variable_label"].values[0]
    subplot.set_xlabel(x_label, fontsize=font_size)
    subplot.set_ylabel(y_label, fontsize=font_size)
    subplot.tick_params(labelsize=font_size)
    subplot.set_xlim([good_data.T_a.min(), good_data.T_a.max()])
    subplot.set_ylim([good_data.P.min(), good_data.P.max()])

def statistical_prepro(df):
    # Estimate the power curve with an iterative median estimation technique.
    nbins = 50
    P_estimated = np.zeros(nbins)
    T_estimated = np.zeros(nbins)
    adjust = 0.001
    Tmax = df.T_a.quantile(1 - adjust)
    Tmin = df.T_a.quantile(0 + adjust)
    dT = (Tmax - Tmin) / nbins
    P_estimated[0] = 0.0

    for i in range(1, nbins):
        Tl = Tmin + i * dT
        Tr = Tl + dT
        # Median of P in the bin i
        T_estimated[i] = (Tl + Tr) / 2.0
        T_subset = df.loc[(df.T_a >= Tl) & (df.T_a < Tr), :]
        # Power distribution of bin i
        Pi = df.loc[T_subset.index, "P"]
        if (len(Pi) > 0):
            P_estimated[i] = statistics.median(Pi)
        else:
            P_estimated[i] = P_estimated[i - 1]
    P_estimated[0] = P_estimated[1]
    # Each point is removed if it is further than 5*sigma from the median at each bin
    preprocessed_df= df.copy()
    for i in range(1, nbins):
        Tl = Tmin + i * dT
        Tr = Tl + dT
        subset = df.loc[(df.T_a >= Tl) & (df.T_a < Tr), :]
        preprocessed_df = preprocessed_df.drop(subset.index)
        Pi = subset.loc[subset.index, "P"]  # power distribution of bin i
        sigma = median_abs_deviation(Pi)
        subset = subset.loc[np.abs(subset.P - P_estimated[i]) < 5 * sigma, :]
        preprocessed_df = pd.concat([preprocessed_df, subset])
        preprocessed_df = preprocessed_df.sort_index()
    return preprocessed_df, P_estimated, T_estimated