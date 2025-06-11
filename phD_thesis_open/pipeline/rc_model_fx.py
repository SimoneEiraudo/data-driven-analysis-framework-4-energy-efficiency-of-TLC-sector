import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import statsmodels
from sklearn import metrics as skmetrics
from statsmodels.tsa.arima.model import ARIMA
import piecewise_regression
import numpy as np

def calc_PUE(P, P_tlc):
    P_base=P_tlc.unique()
    if len(P_base)!=1:
        print("WARNING: the TLC power provided is not unique. P_tlc will be estimated as a mean value")
        P_base=P_tlc.mean()
    PUE=P.sum()/(P_base*len(P))
    return PUE
def ttc_layer(z, layers):
    """
    Calculate the Thermal Time Constant (TTC) for a given layer z.
    :param z: Current layer index (1-based)
    :param layers: List of layer properties [(thickness (m), thermal conductivity (W/(m°C)), mass (kg/m²), specific heat capacity (Whr/kg°C))]
    :return: TTC of layer z (hr)
    """
    r0 = 0.04  # Thermal resistance of the external laminar layer (m²°C/W)
    sum_ri = r0 + sum(layers[i][0]/layers[i][1] for i in range(z)) #calculate resistance of previous layers
    R_t = layers[z][0]/layers[z][1] #calculate thermal resistance of layer Z
    C_t = layers[z][2]*layers[z][3]
    return (sum_ri + 0.5 * R_t) * C_t

def ttc_building_element(layers):
    """
    Calculate the total Thermal Time Constant (TTC) of the building envelope element.
    :param layers: List of layer properties [(thickness (m), thermal conductivity (W/(m°C)), mass (kg/m²), specific heat capacity (Whr/kg°C))]
    :return: Total TTC (hr)

    each layer is defined by 4 parameters: 0: d,
    0 :param d: Thickness of the layer (m)
    1 :param lambd: Specific thermal conductivity of the layer (W/(m°C))
    2 :param m: Mass of the layer (kg/m²)
    3 :param cp: Specific heat capacity of the layer (Whr/kg°C)
    """
    L = len(layers)
    return sum(ttc_layer(z, layers) for z in range(L))

def total_ttc(ttc_opaque, ttc_transparent, wwr, volume, surface_to_volume_ratio):
    """
    Calculate the total Thermal Time Constant (TTC) considering both opaque and transparent elements and air volume.
    :param ttc_opaque: TTC of opaque element (hr)
    :param ttc_transparent: TTC of transparent element (hr)
    :param wwr: Window-to-wall ratio (0-1)
    :param volume: Building volume (m³)
    :param surface_to_volume_ratio: Surface-to-volume ratio (1/m)
    :return: Total TTC (hr)
    """
    air_density = 1.2  # kg/m³
    air_specific_heat = 0.24  # Whr/kg°C
    internal_heat_transfer_coefficient = 8  # W/m²°C

    air_mass = volume * air_density  # Total air mass in kg
    air_thermal_mass = air_mass * air_specific_heat  # Thermal mass in Whr/°C
    air_ttc = air_thermal_mass / (
                internal_heat_transfer_coefficient * surface_to_volume_ratio * volume)  # TTC of air in hr

    return (1 - wwr) * ttc_opaque + wwr * ttc_transparent + air_ttc

def global_heat_transfer_coefficient(layers):
    """
    Calculate the global heat transfer coefficient (U-value) for the building element.
    :param layers: List of layer properties [(thickness (m), thermal conductivity (W/(m°C)), mass (kg/m²), specific heat capacity (Whr/kg°C))]
    :return: U-value (W/m²°C)
    """
    total_resistance = sum(layer[0]/layer[1] for layer in layers) + 0.04
    return 1 / total_resistance

def cop_curve(load, temp, T_in, cop_nom, Qt_nom, T_set=29, a_load=-0.3, b_load=0.7, Cc=0.9, Cd=0.25, autosize=True):
    data = load.copy()
    ## on-off condition
    data['onoff'] = np.where(load > 0, 1, 0)
    ## check cooling system saturation and unmet demand
    if autosize==True:
        Qt_nom = load.nlargest(10).mean()  #autosize cooling system

    data['CR'] = (load / Qt_nom).clip(upper=1)
    data['unmet'] = (load - np.minimum(load, Qt_nom)).clip(lower=0)

    ## calculate COP separately according to load and temperature
    data['cop_load'] = a_load * data['CR'] ** 2 + b_load * data['CR'] + cop_nom  # Empirici, mimano il refr
    T_cnom = T_set  # we hipothize that the cooling system is designed to have the best performance at T_in=T_set - 5
    T_hnom = temp.max() - 5  # and T_ext = 10 degrees lower than the max external temperature of that site
    delta_exchange = 10
    cop_carnot_nominal = (T_cnom + 273.15) / ((T_hnom + 273.15) - (T_cnom + 273.15) + delta_exchange)  # Caso migliore in cui la T_ext è pari a 0 (fissa)
    cop_max = (T_in + 273.15) / ((T_in + delta_exchange + 273.15) - (T_in + 273.15))
    data['cop_carnot'] = np.where(temp > T_in, ((T_in + 273.15) / ((temp + 273.15) - (T_in + 273.15) + delta_exchange)),
                                  cop_max)  # Perché sta +10? Scaling factor ==> si prendono in considerazione gli scambi termici

    ## Calculate a unique COP considering both load and temperature
    data['cop_temp'] = (data['cop_load'] * data['cop_carnot'] / cop_carnot_nominal).clip(lower=1)

    ## adjust COP according to Partial Load Factor
    data['plf'] = 1 / (1 + (Cd * (1 - data['CR'])) / (1 - Cd * (1 - data['CR'])) + (1 - Cc) * (1 - data['CR']) / data['CR'])
    data['cop_pl'] = (data['cop_temp'] * data['plf']).clip(lower=1)
    return data['cop_pl']

def calc_tests(eta):
    # Test diagnostici
    stats = {}

    # Test stationarity
    adf = statsmodels.tsa.stattools.adfuller(eta)
    if (adf[1] <= 0.05) & (adf[4]['5%'] > adf[0]):
        stationarity = 'stationary'
    else:
        stationarity = 'non stationary'
    stats['adfuller'] = (adf[1], adf[4]['5%'], adf[0], stationarity)
    # adf[1] <= 0.05) & (adf[4]['5%'] > adf[0]) => stationary else non stationary

    # Test autocorrelazione lag-1
    # 1-(statsmodels.stats.stattools.durbin_watson(self.df.eta))/2 è una stima del coefficeinte di autocorrelazione
    durbin_watson = statsmodels.stats.stattools.durbin_watson(eta)
    stats['durbin_watson'] = (durbin_watson, "strong lag-1 autorcorrelation" if (durbin_watson < 1.5) | (
            durbin_watson > 2.5) else 'low lag-1 autocorrelation')
    # This statistic will always be between 0 and 4.
    # The closer to 0 the statistic, the more evidence for positive serial correlation.
    # The closer to 4, the more evidence for negative serial correlation.

    # Test di normalità
    jarque_bera = statsmodels.stats.stattools.jarque_bera(eta)[1]
    stats['jarque_bera'] = (jarque_bera, 'normal' if jarque_bera < 0.05 else 'non normal')
    # p-value < 0.05 => non è normale

    # Test eteroschedasticità
    breakvar = statsmodels.tsa.stattools.breakvar_heteroskedasticity_test(resid=eta)[1]
    stats['breakvar'] = (breakvar, 'heteroskedastic' if breakvar < 0.05 else 'homoskedastic')
    # p-value < 0.05 => sono eteroschedastici
    return stats

def calc_metrics(true_values, predict_values):
    metr = {}
    metr['r2'] = skmetrics.r2_score(true_values, predict_values)
    metr['rmse'] = np.sqrt(skmetrics.mean_squared_error(true_values, predict_values))
    metr['mae'] = skmetrics.mean_absolute_error(true_values, predict_values)
    metr['mape'] = skmetrics.mean_absolute_percentage_error(true_values, predict_values)
    metr['ev'] = skmetrics.explained_variance_score(true_values, predict_values)
    return metr

def plot_timeseries(df):
    fig, ax = plt.subplots(5, figsize=(9, 8), sharex=True)
    df['T_ext'].plot(ax=ax[0], label='T_ext')
    ax[0].hlines(df['T_set'], df['T_set'].index[0], df['T_set'].index[-1], color='red', linestyle='--', label='T_sp')
    ax[0].set_ylabel('Temperature [°C]')
    df['Q_cool'].plot(ax=ax[1], label='Q_cool')
    df[['Q_tras', 'Q_gain', 'Q_sol', 'Q_vent']].plot(ax=ax[1],alpha=0.5)
    ax[1].set_ylabel('Heat Flows [W]')
    df['Q_cool'].plot(ax=ax[2], label='Q_cool')
    ax[2].set_ylabel('Total Heat Flow [W]')

    df[['P_cool', 'P_tlc', 'P_diss', 'P_vent']].plot(ax=ax[3])
    ax[3].set_ylabel('Electrical Power [W]')
    df['P'].plot(ax=ax[4], label='P_el')
    ax[4].set_ylabel('Total Electrical Power [W]')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()

def fit_energy_signature(df, n_bkp=1):
    es = piecewise_regression.Fit(df['T_ext'].values, df['P'].values, n_breakpoints=n_bkp)
    es.summary()
    return es

def plot_energy_signature(es,ax=None):
    if ax is None:
        ax = plt.gca()
    # Plot the data, fit, breakpoints and confidence intervals
    es.plot_data(color="grey", s=1,)
    # Pass in standard matplotlib keywords to control any of the plots
    es.plot_fit(color="red", linewidth=2)
    es.plot_breakpoints()
    es.plot_breakpoint_confidence_intervals()
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Electrical Power [W]')
    plt.tight_layout()

def plot_res_diagnostic(res):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    plot_acf(res,ax=ax[0])
    plot_pacf(res,ax=ax[1])
    plt.tight_layout();
    statsmodels.tsa.seasonal_decompose(res).plot();
