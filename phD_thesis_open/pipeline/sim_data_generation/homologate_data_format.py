import pandas as pd
from pipeline.definitions import *
from os import listdir
from os.path import isfile, join

def homologate_dataframe(sources="all"):
    #load ID converters
    real_ID_conv= pd.read_excel(os.path.join(DATASETS, "ID_converter_real.xlsx"))
    sim_ID_conv= pd.read_excel(os.path.join(DATASETS, "ID_converter_sim.xlsx"))

    #first load real data
    real_data= pd.read_csv(os.path.join(REAL_DATA, "real_buildings_dataset"))

    #add anomaly column to real data
    real_data["anomaly"]=-1
    # #change ID
    for ID, newID in zip(real_ID_conv.TIM_name, real_ID_conv.my_names):
        real_data.replace(to_replace=ID, value=newID, inplace=True)

    #add the equivalent temperature to the real world dataset and fill the missing values for the last day of year 22019
    new_real_data=pd.DataFrame()

    for building in real_data.ID.unique():
        df=real_data.loc[real_data.ID==building, :]
        new_df=df.copy()
        new_df['T_eq_3'] = df.T_a.rolling(3, min_periods=1).mean()
        new_df['T_eq_6'] = df.T_a.rolling(6, min_periods=1).mean()
        new_df['T_eq_12'] =df.T_a.rolling(12, min_periods=1).mean()
        new_df['T_eq_24']=df.T_a.rolling(24, min_periods=1).mean()
        new_df.loc[new_df.index[8736:8760],["s_D", "c_D", "s_H", "c_H"]]=df.loc[df.index[8736+8760:8760+8760],["s_D", "c_D", "s_H", "c_H"]].values
        new_df.loc[new_df.index[8736:8760], "dayType"]=0
        new_real_data = pd.concat([new_real_data, new_df])

    #then  load and modify simulated data to be compliant with real ones
    path=SIM_DATA_NO_FORMAT
    files= [f for f in listdir(path) if isfile(join(path, f))]
    sim_data=pd.DataFrame()

    for file in files:
        df=pd.DataFrame()
        real_df=new_real_data.loc[new_real_data.ID==new_real_data.ID.unique()[0], :]
        gen_df=pd.read_csv(path+"/"+file)
        #get data from simulations
        df['T_a']=gen_df.T_ext
        df['P']=gen_df.P
        df['G']=gen_df.I_sol
        df['anomaly']=gen_df.anomaly

        #add additional columns as it was done for real data
        df['T_b']=df.T_a
        df['T_eq_3'] = df.T_a.rolling(3, min_periods=1).mean()
        df['T_eq_6'] = df.T_a.rolling(6, min_periods=1).mean()
        df['T_eq_12'] = df.T_a.rolling(12, min_periods=1).mean()
        df['T_eq_24']=df.T_a.rolling(24, min_periods=1).mean()

        #fill unknown variables
        df[['DP', 'RH', 'Wv', 'Wgv', 'atmP', 'UV', 's_Wa','c_Wa']]=0

        #fill calendar info from real data
        df[['s_H', 'c_H', 'dayType', 's_D', 'c_D']]=real_df[['s_H', 'c_H', 'dayType', 's_D', 'c_D']]

        #fill ID
        df["ID"]=sim_ID_conv.loc[sim_ID_conv.generator_names==file, "my_names"].values[0]

        sim_data=pd.concat([sim_data, df])

    #reorder columns
    sim_data = sim_data[new_real_data.columns]
    if sources=="all":
        all_data=pd.concat([sim_data, new_real_data])
    elif sources=="simulations":
        all_data=sim_data.copy()

    all_data=all_data.reset_index()
    all_data=all_data.drop("index", axis=1)
    return all_data

if __name__ == "__main__":
    all_data=homologate_dataframe(sources="simulations") #or "all"
    all_data.to_csv(os.path.join(DATA_RAW, "all_buildings_dataset"), index=False)
