import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join

from pipeline.definitions import *

def cusum_tab(x_setup, x_monitor=None, k=0.5, h=11, moving_range=False):
    if x_monitor is None:
        x = x_setup
    else:
        x = np.concatenate([x_setup, x_monitor])

    mu, sigma = x_setup.mean(), x_setup.std()
    #print(mu, sigma)
    if moving_range:
        MR = np.zeros(len(x) - 1)
        for n in range(len(MR)):
            MR[n] = np.abs(x[n + 1] - x[n])
        sigma = MR.mean() / 1.128

    Cpos = np.zeros(len(x))
    Cneg = np.zeros(len(x))
    for n in range(1, len(x)):
        Cpos[n] = max(0, (x[n] - mu) - k * sigma + Cpos[n - 1])
        Cneg[n] = max(0, -k * sigma - (x[n] - mu) + Cneg[n - 1])
    anomaly_threshold=sigma*h
    return Cpos, Cneg, anomaly_threshold,  sigma

def plot_cusum_tab(Cpos, Cneg, anomaly_threshold, sigma=1, ax=None, k=0.5):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(Cpos, label=r"$C_+$")
    ax.plot(Cneg, label=r"$C_-$")
    ax.hlines(y=k * sigma, xmin=Cpos.index[0], xmax=Cpos.index[-1], ls="-.", color='green', label="K")
    ax.hlines(y=anomaly_threshold, xmin=Cpos.index[0], xmax=Cpos.index[-1], ls="-.", color='red', label="H")
    ax.legend()
    ax.set_xlabel('Date')
    # ax.set_xticks(ax.get_xticks())
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    #ax.set_xticklabels([])
    ax.grid()
    #plt.savefig("test")
    pass

def calculate_confusion_matrix(dataset, IDs, k=0.5, h=11):
    confusion_matrix = {}
    det_delay = pd.DataFrame(index=["delay"])
    total_con_mat = pd.DataFrame([[0, 0], [0, 0]], index=["Abnormal", "Conform"],
                                 columns=["Abnormal", "Conform"])  # , columns.name="Actual")
    total_con_mat.index.name = "Predicted"
    total_con_mat.columns.name = "Actual"
    dataset.loc[:, "detected_anomaly"] = 0

    for caseStudy in IDs:
        con_mat = total_con_mat.copy()
        for col in con_mat.columns:
            con_mat[col].values[:] = 0

        df = dataset.loc[dataset.ID == caseStudy, :]
        df = df.reset_index()
        df["detected_anomaly"] = 0
        res = df["residuals_semiPar"]
        norm_res = res / df.P
        _, _, res_setup, res_monitor = train_test_split(norm_res, norm_res, test_size=0.33, shuffle=False)
        Cpos, Cneg, anomaly_threshold, sigma = cusum_tab(res_setup, x_monitor=res_monitor, k=k, h=h, moving_range=False)
        Cpos = pd.Series(Cpos)
        Cneg = pd.Series(Cneg)

        index_det_anom = (Cpos > anomaly_threshold) | (Cneg > anomaly_threshold)
        df.loc[index_det_anom, "detected_anomaly"] = 1
        # get start and end of real and detected anomalies
        real_anom_location = locate_anomaly(df, "anomaly")
        det_anom_location = locate_anomaly(df, "detected_anomaly")

        for start, end in zip(real_anom_location["first"].values, real_anom_location["last"].values):
            det_delay[caseStudy]=np.nan
            if ((start <= det_anom_location["first"].values) & (end > det_anom_location["first"].values)).any():
                con_mat.loc["Abnormal", "Abnormal"] = con_mat.loc["Abnormal", "Abnormal"] + 1
                det_anom_start_ind = det_anom_location.loc[
                                     ((start <= det_anom_location["first"]) & (end > det_anom_location["first"])),
                                     :].index.values
                det_anom_start = det_anom_location.loc[det_anom_start_ind, "first"].min()
                det_anom_location = det_anom_location.drop(det_anom_start_ind)
                det_delay[caseStudy] = det_anom_start - start

            else:
                con_mat.loc["Conform", "Abnormal"] = con_mat.loc["Abnormal", "Abnormal"] + 1

        con_mat.loc["Abnormal", "Conform"] = det_anom_location.shape[0]
        confusion_matrix[caseStudy] = con_mat
        total_con_mat = total_con_mat + con_mat
        dataset.loc[dataset.ID==caseStudy, "detected_anomaly"]=df["detected_anomaly"].values

    return dataset, confusion_matrix, total_con_mat, det_delay

def anomaly_detection_score(confusion_matrix, w_sensitivity=0.5):
    w_precision = 1 - w_sensitivity
    sensitivity = sensitivity_from_matrix(confusion_matrix)
    precision = precision_from_matrix(confusion_matrix)
    anomaly_score = sensitivity * w_sensitivity + precision * w_precision
    return anomaly_score

def f1_from_matrix(confusion_matrix):
    TP = confusion_matrix.loc["Abnormal", "Abnormal"]
    FN = confusion_matrix.loc["Conform", "Abnormal"]
    FP = confusion_matrix.loc["Abnormal", "Conform"]
    f1 = 2 * TP / (2 * TP + FP + FN)
    return f1


def locate_anomaly(df, col, threshold=3):
    out = []
    m = df[col].eq(1)
    g = (df[col] != df[col].shift()).cumsum()[m]
    mask = g.groupby(g).transform('count').ge(threshold)
    filt = g[mask].reset_index()
    output = filt.groupby(col)['index'].agg(['first', 'last'])
    output.insert(0, 'col', col)
    out.append(output)

    return pd.concat(out, ignore_index=True)

def precision_from_matrix(confusion_matrix):
    TP = confusion_matrix.loc["Abnormal", "Abnormal"]
    FN = confusion_matrix.loc["Conform", "Abnormal"]
    FP = confusion_matrix.loc["Abnormal", "Conform"]
    f1 = TP / (TP + FP)
    return f1

def sensitivity_from_matrix(confusion_matrix):
    TP = confusion_matrix.loc["Abnormal", "Abnormal"]
    FN = confusion_matrix.loc["Conform", "Abnormal"]
    FP = confusion_matrix.loc["Abnormal", "Conform"]
    f1 = TP / (TP + FN)
    return f1

def locate_in_df(df, value):
    a = df.to_numpy()
    row = np.where(a == value)[0][0]
    col = np.where(a == value)[1][0]
    return row, col

def plot_residuals(res, temp, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    ax[0].scatter(temp, res, color="gray", s=2)
    ax[0].set_xlabel("Temperature [°C]")
    ax[0].set_ylabel("Residuals")
    ax[0].grid()
    ax[1].plot(res)
    ax[1].set_xlabel("Time step [h]")
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    ax[1].grid()
    pass

def retrive_data_from_generator(caseStudy):
    #load original data from generator to check anomalies
    path=SIM_DATA_NO_FORMAT
    datasets = [f for f in listdir(path) if isfile(join(path, f)) and "simData" in f]
    buildings=list(set([f.split("_")[2] for f in datasets]))
    weathers=list(set([f.split("_")[3] for f in datasets]))
    anomalies=list(set([f.split("_")[6] for f in datasets if "anomaly" in f]))

    #load all the datasets in a dictionary
    generator_dfs={}
    for building in buildings:
        for weather in weathers:
            file="simData_building_"+building+"_"+weather+"_weather"
            if file in datasets:
                generator_dfs[building, weather, "original"]=pd.read_csv(path+"/"+file)
            for anomaly in anomalies:
                file="simData_building_"+building+"_"+weather+"_weather_anomaly_"+anomaly
                if file in datasets:
                    generator_dfs[building, weather, "anomaly_"+anomaly]=pd.read_csv(path+"/"+file)

    #load dataset description
    sim_ID_conv= pd.read_excel(os.path.join(DATASETS, "ID_converter_sim.xlsx"))


    building=sim_ID_conv.loc[sim_ID_conv.my_names==caseStudy, "generator_names"].values[0].split("_")[2]
    weather=sim_ID_conv.loc[sim_ID_conv.my_names==caseStudy, "generator_names"].values[0].split("_")[3]
    anomaly=sim_ID_conv.loc[sim_ID_conv.my_names==caseStudy, "generator_names"].values[0].split("_")[6]
    keys=[building, weather, "anomaly_"+anomaly]
    generator_df=generator_dfs[keys[0], keys[1], keys[2]]
    return generator_df