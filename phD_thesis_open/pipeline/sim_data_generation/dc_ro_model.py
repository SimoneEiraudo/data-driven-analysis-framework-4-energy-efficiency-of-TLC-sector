"""
Modello semi-stazionario
Dinamica semplice con costante di tempo
ventilazione base senza free cooling
consumo elettrico lineare (no COP)
no sottomodelli
white noise N(0,sigma), autocorrelazione e correlazione con temperatura
"""
import random
import pandas as pd
import numpy as np
from pipeline.definitions import *
from pipeline.rc_model_fx import ttc_building_element, total_ttc, global_heat_transfer_coefficient, cop_curve, fit_energy_signature, plot_timeseries, plot_energy_signature

class DataCenter():
    def __init__(self,
                 weather_file: str,
                 years=None,
                 geometries=None,
                 physics=None,
                 walls_layers=None,
                 windows_layers=None,
                 T_sp: float = 29,  # °C
                 gain_TLC_density: float = None,  # W
                 free_cooling: bool = True,
                 ventilation: float = 2,  # 1/h numero di ricambi orari, 0 per no ventilazione
                 cop: bool = True,
                 cop_params=None,
                 noise=None,
                 anomaly=None,
                 retrofit=None):

        # Default parameters
        if years is None:
            years = [2018, 2019]
        if noise is None:
            noise = {'sigma_scale': 50, 'rho': 0.5, 'n_h': 2} #più è alto sigma, più riduce il rumore
        if cop_params is None:
            cop_params = {'cop_nom': 4, 'Qt_nom': 100000, "T_set":29,  'a_load': -0,
                          'b_load': 0, 'Cc': 0.90, 'Cd': 0.1}
        if physics is None:
            physics = {'cp_au': 1006/3600, # Wh/kg*K aria umida
                       'rho_au': 1.2} # kg/m3 aria umida}
        if geometries is None:
            geometries = {'V': 950 * 3,
                          'SV_ratio': 0.75,
                          'win2wal_ratio': 0.1,
                          'wallsun_ratio': 0.01}

        if walls_layers is None:
            s_concr = 0.2
            s_ins = 0.02
            rho_concr=2076
            rho_ins=18
            layers_opaque = [# (thickness (m), thermal conductivity (W/(m°C)), mass (kg/m²), specific heat capacity (Whr/kg°C))
                (s_concr, 1.37,  rho_concr* s_concr, 1826 / (rho_concr * 3.6)),  # concrete (rho=2076 kg/m^3, cp=1826 kJ/(m^3*K))
                (s_ins, 0.038,  rho_ins* s_ins, 1500 / (rho_ins* 3.6))  ]         # EPS insulation layer (rho=18 kg/m^3, cp=1500 kJ/(m^3*K))

        #retrofit A must be implemented here, while the others retrofit, as well as anomaly, are implemented later on
        if retrofit=="A":
            s_ins=0
            layers_opaque[1]=(s_ins, 0.038,  rho_ins* s_ins, 1500 / (rho_ins* 3.6))

        if windows_layers is None:
            s_glass = 0.006
            s_air = 0.03
            rho_glass=2300
            rho_air=1.3
            layers_transparent = [ # (thickness (m), thermal conductivity (W/(m°C)), mass (kg/m²), specific heat capacity (Whr/kg°C))
                (s_glass, 1.05,  rho_glass * s_glass, 836 / (rho_glass * 3.6)),
                (s_air, 5.56, rho_air * s_air, 1.004 / 3.6),
                (s_glass, 1.05, rho_glass * s_glass, 836 / (rho_glass * 3.6)) ]


        # Import weather file
        temp_data = pd.read_csv(weather_file)  # (r'Timeseries_38.111_13.352_SA2_90deg_0deg_2018_2020.csv')
        temp_data.index = pd.to_datetime(temp_data.time, format='%Y%m%d:%H%M')
        # Selezione anno
        self.df = temp_data.loc[np.isin(temp_data.index.year, years), ['T2m', 'G(i)']]
        self.df = self.df.rename(columns={'T2m': 'T_ext', 'G(i)': 'I_sol'})

        # Proprietà geometriche
        for key, value in geometries.items():
            self.__setattr__(key, value)
        self.A_disp = self.V * self.SV_ratio
        self.A_floor= self.A_disp/6

        # Proprietà fisiche
        for key, value in physics.items():
            self.__setattr__(key, value)

        #calculate thermal capacity
        self.air_thermal_capacity = self.rho_au * self.cp_au * self.V  # Wh/K
        self.walls_thermal_capacity = sum([layer[2]*layer[3] for layer in layers_opaque])*self.A_disp*(1-self.win2wal_ratio)
        self.windows_thermal_capacity = sum([layer[2] * layer[3] for layer in layers_transparent]) * self.A_disp * (self.win2wal_ratio)
        self.thermal_capacity=self.air_thermal_capacity+(self.walls_thermal_capacity+self.windows_thermal_capacity)/2

        #calculate U
        self.U_wall= global_heat_transfer_coefficient(layers_opaque)
        self.U_windows= global_heat_transfer_coefficient(layers_transparent)

        #calculate constant of time
        ttc_opaque = ttc_building_element(layers_opaque)
        ttc_transparent = ttc_building_element(layers_transparent)
        ttc_total = total_ttc(ttc_opaque, ttc_transparent, self.win2wal_ratio, self.V, self.SV_ratio)
        self.tau = ttc_total  # h costante di tempo termica
        self.time_lag = int(np.ceil(self.tau))

        # Temperatura di setpoint
        self.T_sp = T_sp
        self.df['T_set'] = self.T_sp * np.ones(self.df.shape[0])
        # Temperatura interna reale (<=set point)
        self.df['T_in']=self.df['T_set'] #inizializza la T_in=T_set, poi aggiorna nel ciclo di bilancio termico

        #inizialize building without any anomaly and retrofit
        self.df["anomaly"]=0
        self.df["retrofit"] = 0
        self.df['cop_anom_degr']=0

        #inizialize free cooling and cooler status
        if free_cooling:
            self.df['free_cooling']=1
        else:
            self.df['free_cooling'] = 0

        self.df["cooler"]=1

        ##Definire flussi indipendenti da T_in (fuori da ciclo)
        # Gain interni TLC
        gain_TLC=gain_TLC_density*self.A_floor
        self.df['Q_gain'] = gain_TLC # Server
        # Consumi elettrici TLC
        self.df['P_tlc'] = self.df['Q_gain'] / 0.98
        self.df['P_diss'] = self.df['P_tlc'] * 0.10  #AC/DC conversion and losses

        # Gain solare
        self.SHGC = 0.5  # coefficiente di trasmissione solare superfici traspqrenti solar heat gain coefficient
        self.alpha_sol = 0.7  # assorbimento superfici opache
        self.tau_sol = 0.1  # trasmissione superfici opache
        Q_trasp = self.A_disp * self.win2wal_ratio * self.wallsun_ratio * self.df[
            'I_sol'] * self.SHGC  # I_sol su pareti verticali
        Q_opaco = self.A_disp * (1 - self.win2wal_ratio) * self.wallsun_ratio * self.df[
            'I_sol'] * self.alpha_sol * self.tau_sol
        self.df['Q_sol'] = (Q_trasp + Q_opaco)

        ##Definire flussi variabili (parametri fuori ciclo, aggiornamento valori dentro ciclo)
        # Ventilazione
        self.ventilation = ventilation  # ventilation 1/h numero di ricambi orari 0 per no ventilazione
        self.m_au = self.ventilation * self.V * self.rho_au  # kg/h
        self.deltap = 100 + 100 + 50 + 30  # Pa stima condotti, filtri, griglie, diffusori/accessori
        self.vent_eff = 0.85
        # Calcolo Degree Hour: DH < 0 Free Cooling, DH > 0 Cooling
        self.df['DH_stz'] = self.df['T_ext'] - self.df['T_set']
        self.df['DH'] = self.df['DH_stz']

        #generazione anomalie
        if anomaly:
            #definire una maschera randomica (durata+tempo) per l'evento anomalo
            #durata
            if anomaly in ["A", "B", "C", "D"]:
                d_min, d_max=12, 24*7*4
            elif anomaly=="E":
                d_min, d_max=2, 24

            d=random.randint(d_min, d_max)

            #momento di inizio
            start_t=random.randint(8760+3000, 8760*2-d-2000) ####8760 per spostare l'anomalia sul secondo anno,l'inizio a partire da 3000 e la fine a 2000 step in meno per avere l'anomalia sull'estate
            self.df["anomaly"][start_t:start_t+d]=1

            if anomaly=="A": #corrisponde all'anomalia "malfunzionamento macchina frigorifera"
                degr_percentage_min = 5
                degr_percentage_max = 30
                degr_percentage = random.randint(degr_percentage_min, degr_percentage_max)
                self.df.loc[self.df.anomaly == 1, 'cop_anom_degr'] = degr_percentage/100

            elif anomaly=="B": #corrisponde all'anomalia "manutenzione rack"
                rack_percentage_min=1
                rack_percentage_max=15
                rack_percentage=random.randint(rack_percentage_min, rack_percentage_max)
                self.df.loc[self.df.anomaly==1, 'Q_gain'] = gain_TLC*(100-rack_percentage)/100
                #recalculate P_tlc and P_diss
                self.df.loc[self.df.anomaly==1,'P_tlc'] = self.df['Q_gain'] / 0.98
                self.df.loc[self.df.anomaly==1,'P_diss'] = self.df['P_tlc'] * 0.10

            elif anomaly=="C":   #corrisponde all'anomalia "tasto comfort ON"
                T_comfort=24
                self.df.loc[self.df.anomaly == 1, 'T_set']=T_comfort

            elif anomaly=="D":   #corrisponde all'anomalia "free cooling rotto"
                self.df.loc[self.df.anomaly == 1, 'free_cooling']=0

            elif anomaly=="E":  #corrisponde all'anomalia "grave malfunzionamento macchina frigorifera"
                self.df.loc[self.df.anomaly == 1, 'cooler'] = 0

        if retrofit:
            self.df.retrofit=1
            if retrofit=="A": #riduzione isolamento involucro
                None #Notice: the modifications for this retrofit are already implemented at the beginnning (building layers definition)
            elif retrofit=="B":  #Riduzione assorbimento solare involucro
                self.alpha_sol = 0.15  # assorbimento superfici opache
                self.df["Q_trasp"] = self.A_disp * self.win2wal_ratio * self.wallsun_ratio * self.df['I_sol'] * self.SHGC  # I_sol su pareti verticali
            elif retrofit=="C": #Innalzamento set point
                new_setpoint = 31
                self.df.loc[self.df.retrofit == 1, 'T_set'] = new_setpoint
                self.df['T_in'] = self.df['T_set']
            elif retrofit=="D": #Sostituzione macchina frigorifera
                cop_params = {'cop_nom': 5.5, 'Qt_nom': 100000, "T_set": 29, 'a_load': -0,
                              'b_load': 0, 'Cc': 0.99, 'Cd': 0.05}
            elif retrofit=="E": #Decommissioning
                rack_percentage = 10
                self.df['Q_gain'] = gain_TLC * (100 - rack_percentage) / 100
                # recalculate P_tlc and P_diss
                self.df['P_tlc'] = self.df['Q_gain'] / 0.98
                self.df['P_diss'] = self.df['P_tlc'] * 0.10

        #Set initial conditions
        dt = 1
        self.df.loc[str(self.df.index[0]), 'T_in']=self.df.loc[str(self.df.index[0]), 'T_set']
        self.df.loc[str(self.df.index[0]), 'DH_tau'] = self.df.loc[str(self.df.index[0]), 'T_ext'] - self.df.loc[str(self.df.index[0]), 'T_in']
        self.df.loc[str(self.df.index[0]), 'DH']=self.df.loc[str(self.df.index[0]), 'DH_tau']
        self.df.loc[str(self.df.index[0]), 'CDH']=self.df.loc[str(self.df.index[0]), 'DH_tau']
        self.df.loc[str(self.df.index[0]), "Q_tras"]=(self.U_wall * self.A_disp * (1 - self.win2wal_ratio) + self.U_windows * self.A_disp * self.win2wal_ratio) * self.df.loc[str(self.df.index[0]), 'DH']
        self.df.loc[str(self.df.index[0]), 'Q_vent'] = min(0, self.m_au * self.cp_au * self.df.loc[str(self.df.index[0]), 'DH']) * self.df.loc[str(self.df.index[0]), "free_cooling"]  # < 0  kg/h * Wh/kg*K * K #todo: deltaT for ventilation should be istantaneous
        self.df.loc[str(self.df.index[0]), 'P_vent'] = self.ventilation * self.V * self.deltap / (self.vent_eff * 3600) * self.df.loc[str(self.df.index[0]), "free_cooling"]  # W
        self.df.loc[str(self.df.index[0]), 'P_diss'] = self.df.loc[str(self.df.index[0]), 'P_diss'] + self.df.loc[str(self.df.index[0]), 'P_vent'] * 0.10  # TODO: why to include ventilation here??
        self.df.loc[str(self.df.index[0]), "Q_tot"] = self.df.loc[str(self.df.index[0]), 'Q_tras'] + self.df.loc[str(self.df.index[0]), 'Q_gain'] + self.df.loc[str(self.df.index[0]), 'Q_sol'] + self.df.loc[str(self.df.index[0]), 'Q_vent']
        self.df.loc[str(self.df.index[0]), "Q_cool"] = max(self.df.loc[str(self.df.index[0]), "Q_tot"], 0)  #assumiamo che il condizionamento sta bilanciando perfettamente la somma dei flussi  al fine di mantenere il set point
        self.df.loc[str(self.df.index[0]), "P_vent"] = 0

        for t in range(1, self.df.shape[0]):
            #calculate deltaT (in-out)
            deltaT = self.df.loc[str(self.df.index[t-1]), 'T_ext'] - self.df.loc[str(self.df.index[t-1]), 'T_in']
            deltaTtau =  (self.tau) / (self.tau + dt) * deltaT
            self.df.loc[str(self.df.index[t]), 'DH_tau'] = deltaTtau
            self.df.loc[str(self.df.index[t]), 'DH']=self.df.loc[str(self.df.index[t]), 'DH_tau']

            # Dispersione termica superfici opache e vetrate
            self.df.loc[str(self.df.index[t]), 'Q_tras'] = (self.U_wall * self.A_disp * (
                        1 - self.win2wal_ratio) + self.U_windows * self.A_disp * self.win2wal_ratio) * self.df.loc[str(self.df.index[t]), 'DH']

            if free_cooling:
                self.df.loc[str(self.df.index[t]), 'Q_vent'] = min(0, self.m_au * self.cp_au * deltaT *self.df.loc[str(self.df.index[t]), "free_cooling"]) # < 0  kg/h * Wh/kg*K * K #todo: deltaT for ventilation should be istantaneous
                self.df.loc[str(self.df.index[t]), 'P_vent'] = self.ventilation * self.V * self.deltap / (self.vent_eff*3600)*self.df.loc[str(self.df.index[t]), "free_cooling"]  # W
                self.df.loc[str(self.df.index[t]), 'P_diss'] = self.df.loc[str(self.df.index[t]), 'P_diss'] + self.df.loc[str(self.df.index[t]), 'P_vent'] * 0.10  # TODO: why to include ventilation here??

            else:
                self.df.loc[str(self.df.index[t]), 'Q_vent'] = self.m_au * self.cp_au * self.df.loc[str(self.df.index[t]), 'DH']  # kg/h * Wh/kg*K * K
                self.df.loc[str(self.df.index[t]), 'P_vent'] = self.ventilation * self.V * self.deltap / (self.vent_eff*3600)  # W
                self.df.loc[str(self.df.index[t]), 'P_diss'] = self.df.loc[str(self.df.index[t]), 'P_diss'] + self.df.loc[str(self.df.index[t]), 'P_vent'] * 1.10  # TODO: why to include ventilation here??

            #finally calculate the sum of heat fluxes
            self.df.loc[str(self.df.index[t]), 'Q_tot'] = self.df.loc[str(self.df.index[t]), 'Q_tras'] + self.df.loc[str(self.df.index[t]), 'Q_gain'] + self.df.loc[str(self.df.index[t]), 'Q_sol'] + self.df.loc[str(self.df.index[t]), 'Q_vent']

            # if Q_tot<0--> T_in diminuisce
            deltaT_nextStep=self.df.loc[str(self.df.index[t]), 'Q_tot']/self.thermal_capacity

            self.df.loc[str(self.df.index[t]), 'T_in']=self.df.loc[str(self.df.index[t-1]), 'T_in']+deltaT_nextStep
            heat_surplus=0
            # if Q_tot>0, T_in aumenta quanto può
            if self.df.loc[str(self.df.index[t]), 'T_in']>self.df.loc[str(self.df.index[t]), 'T_set']:
                temperature_surplus=self.df.loc[str(self.df.index[t]), 'T_in']-self.df.loc[str(self.df.index[t]), 'T_set']

                if self.df.loc[str(self.df.index[t]), 'cooler'] == 1:  # the cooler is available, it will provide cooling to face the heat surplus
                    heat_surplus = temperature_surplus * self.thermal_capacity
                    self.df.loc[str(self.df.index[t]), 'T_in']=self.df.loc[str(self.df.index[t]), 'T_set'] #the temperature is brought down to T_set

            self.df.loc[str(self.df.index[t]), 'Q_cool'] = heat_surplus


        self.df['DH_tau'][0:self.time_lag + 1] = self.df['DH_tau'][self.time_lag + 2]

        # Cooling Degree Hour
        self.df['CDH'] = self.df['DH'].clip(lower=0)  # > 0

        # aggiunta errori con white noise N(0,sigma), autocorrelazione e correlazione con temperatura
        for key, value in noise.items():
            self.__setattr__(key, value)
        self.sigma_eta = (self.df['Q_tot'].clip(lower=0) + self.df['P_tlc'] + self.df[
            'P_diss']+ self.df['P_vent']).std() / self.sigma_scale  # dimensione Q_tot

        self.eta = np.random.normal(0, self.sigma_eta, self.df.shape[0])

        self.epsilon = np.zeros(self.df.shape[0])
        self.epsilon_ar = np.zeros(self.df.shape[0])
        self.epsilon_h = np.zeros(self.df.shape[0])

        self.t_ext_norm = (self.df['DH'] - self.df['DH'].min()) / (self.df['DH'].max() - self.df['DH'].min())

        self.epsilon_h[0] = np.random.normal(0, self.sigma_eta * self.n_h * self.t_ext_norm[0])
        self.epsilon_ar[0] = self.epsilon[0]
        self.epsilon[0] = self.epsilon_ar[0] + self.epsilon_h[0]
        for n in range(1, self.df.shape[0]):
            self.epsilon_ar[n] = self.rho * self.epsilon_ar[n - 1] + self.eta[n]
            self.epsilon_h[n] = np.random.normal(0, self.sigma_eta * self.n_h * self.t_ext_norm[n])
            self.epsilon[n] = self.epsilon_ar[n] + self.epsilon_h[n]

        self.df['eta'] = self.eta
        self.df['epsilon'] = self.epsilon
        self.df['epsilon_ar'] = self.epsilon_ar
        self.df['epsilon_h'] = self.epsilon_h

        # Cooling
        #self.df['Q_cool'] = self.df['Q_tot'].clip(lower=0)  # > 0

        # Ipotesi P cooling lineare o COPcarnot
        if cop:
            self.df['cop'] =  cop_curve(self.df['Q_cool'], self.df['T_ext'], self.df['T_in'],
                                                              **cop_params)
            self.df['cop'] = self.df['cop'] * (1 - self.df['cop_anom_degr'])
            self.df['P_cool'] = (self.df['Q_cool'] /self.df['cop']).fillna(0)
        else:
            fixed_COP=3.5
            self.df['cop']=fixed_COP
            self.df['cop'] = self.df['cop'] * (1 - self.df['cop_anom_degr'])
            self.df['P_cool'] = self.df['Q_cool']/fixed_COP

        # Potenza elettrica totale
        self.df['P'] = (self.df['P_cool'] + self.df['P_tlc'] + self.df['P_diss'] + self.df['P_vent'] + self.df['epsilon']).clip(lower=0)

        # Infine, correggi le "anomalie asintomatiche"
        self.df['real_anomaly']=self.df['anomaly']
        if anomaly in ["A", "D", "E"]:
            self.df.loc[self.df.P_cool<=0, 'real_anomaly']=0

if __name__ == "__main__":

    for tlc_density in ["low"]:#, "high"]:
        for weather in ["cold"]:#, "hot"]:
            for anomaly in [None, "A", "B", "C", "D", "E"]:
                for retrofit in [None, "A"]:#, "B", "C", "D", "E"]:
                    #just run simulation if there is max 1 retrofit or anomaly
                    if (anomaly==None) or (retrofit==None):

                        if weather=="hot":
                            #weather_file = r'C:\Users\simone.eiraudo\PycharmProjects\phD_thesis\data\weather\reference_weathers\palermo_Timeseries_38.111_13.352_SA3_90deg_0deg_2005_2020.csv'
                            weather_file=os.path.join(REF_WEATHERS, "palermo_Timeseries_38.111_13.352_SA3_90deg_0deg_2005_2020.csv")
                        elif weather=="cold":
                            #weather_file=r'C:\Users\simone.eiraudo\PycharmProjects\phD_thesis\data\weather\reference_weathers\torino_Timeseries_45.062_7.668_SA2_90deg_0deg_2005_2020.csv'
                            weather_file = os.path.join(REF_WEATHERS,"torino_Timeseries_45.062_7.668_SA2_90deg_0deg_2005_2020.csv")

                        if tlc_density=="low":
                            gain_TLC_density=215 #W/m^2
                            building_ID = "building_A"
                        elif tlc_density=="high":
                            gain_TLC_density=430 #W/m^2
                            building_ID = "building_B"

                        dc = DataCenter(weather_file=weather_file, retrofit=retrofit, anomaly=anomaly, years=[2018, 2019], gain_TLC_density=gain_TLC_density)
                        df = dc.df

                        #es = fit_energy_signature(df)
                        #plot_timeseries(df)
                        #plot_energy_signature(es)

                        if anomaly:
                            name="simData_"+building_ID+"_"+weather+"_weather_anomaly_"+anomaly
                        elif retrofit:
                            name = "simData_"+building_ID+"_"+weather+"_weather_retrofit_"+retrofit
                        else:
                            name="simData_"+building_ID+"_"+weather+"_weather"


                        df.to_csv(SIM_DATA_NO_FORMAT+"/"+name, index=False)