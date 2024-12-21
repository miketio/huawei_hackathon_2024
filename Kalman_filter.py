
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
import math as m
import pandas as pd

def get_OCV_data(file_name, dataframes=None):
    """
    Retrieves and processes OCV data. If a file path is provided, it reads the .xlsx file. 
    If a preloaded dataframe dictionary is provided, it retrieves the relevant data.
    """
    # Check if file_name is a path or a key in dataframes
    if dataframes is None or file_name not in dataframes:
        # Load the data from an .xlsx file if dataframes dictionary is not provided or file_name is a path
        ocv_df = pd.read_excel(file_name)
    else:
        # Otherwise, retrieve the DataFrame from the provided dataframes dictionary
        ocv_df = dataframes[file_name]
    
    # Process OCV data as before
    ocv_df_cleaned = ocv_df.iloc[1:].reset_index(drop=True)
    
    # Extract charging and discharging data, skipping the first row
    charging_data = ocv_df_cleaned.iloc[:, [1, 2]].dropna().reset_index(drop=True)
    discharging_data = ocv_df_cleaned.iloc[:, [4, 5]].dropna().reset_index(drop=True)

    # Rename columns for clarity
    charging_data.columns = ['data_SOC', 'data_U']
    discharging_data.columns = ['data_SOC', 'data_U']

    # Create the structured dictionary with processed data
    data_OCV = {
        'charge': {
            'SOC': np.array(charging_data['data_SOC']).astype(np.float64),
            'U': np.array(charging_data['data_U']).astype(np.float64)
        },
        'discharge': {
            'SOC': np.array(discharging_data['data_SOC']).astype(np.float64),
            'U': np.array(discharging_data['data_U']).astype(np.float64)
        }
    }
    
    return data_OCV


def soc_to_voltage(data_OCV):
    """
    Defines the interpolated U_ocp(SoC) functions for charging and discharging process
    """
    # Interpolation functions for charge and discharge using OCV data and also inverted ones
    interp_func_charge = interp1d(data_OCV['charge']['SOC'], data_OCV['charge']['U'], 
                                    bounds_error=False, fill_value="extrapolate")
    interp_func_discharge = interp1d(data_OCV['discharge']['SOC'], data_OCV['discharge']['U'], 
                                        bounds_error=False, fill_value="extrapolate")

    return interp_func_charge, interp_func_discharge

data_OCV = get_OCV_data("Cha_Dis_OCV_SOC_Data.xlsx")
ocp_charge_fn, ocp_discharge_fn = soc_to_voltage(data_OCV)

def ocp_charge_derivative(soc):
    return derivative(ocp_charge_fn, soc, dx=1e-6)  # Numerical derivative

def ocp_discharge_derivative(soc):
    return derivative(ocp_discharge_fn, soc, dx=1e-6)  # Numerical derivative

class KalmanFilter(object):
    def __init__(self, x, R0, F, B, P, Q, R, Hx_charge, Hx_discharge, HJacobian_charge, HJacobian_discharge):
        self._x = x
        self._R0 = R0
        self._F = F
        self._B = B
        self._P = P
        self._Q = Q
        self._R = R
        self._Hx_charge = Hx_charge
        self._Hx_discharge = Hx_discharge
        self._HJacobian_charge = HJacobian_charge
        self._HJacobian_discharge = HJacobian_discharge
        self.mode = "charge"  # Default mode is charging

    def set_mode_from_current(self, current):
        """
        Set the mode ('charge' or 'discharge') based on the current value.
        Positive current is charge, negative is discharge.
        """
        if current >= 0:
            self.mode = "charge"
        elif current < 0:
            self.mode = "discharge"
        else:
            self.mode = None  # Undefined for zero current, but you can handle it as needed


    def update(self, z, u=0):
        # Dynamically set mode based on the current sign
        self.set_mode_from_current(u)
        P = self._P
        R = self._R
        x = self._x
        z = z - self._R0 * u
        if self.mode == "charge":
            H = self._HJacobian_charge(self._x)
            hx = self._Hx_charge(self._x)
        elif self.mode == "discharge":
            H = self._HJacobian_discharge(self._x)
            hx = self._Hx_discharge(self._x)

        S = H * P * H.T + R
        K = P * H.T * S.I
        self._K = K

        y = np.subtract(z, hx)
        self._x = x + K * y

        KH = K * H
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P #* I_KH.T + K * R * K.T # maybe not correct. should be I_KH * P

    def predict(self, u=0):
        self._x = self._F * self._x + self._B * u
        self._P = self._F * self._P * self._F.T + self._Q

    @property
    def x(self):
        return self._x


def get_EKF(Q_tot = 10094,time_step = 1, R0 = 0.0010171792063898184, R1 = 0.00023313610232415333, R1C1 = 10.238807570373595, 
            R2 = 2.8594334855040053e-06, R2C2 = 19.70943058930129, R = 1.0472922011993644, Q11 = 0.9637489783150914, 
            Q12 = 2.864721218474134, Q13 = 0.19637595723867785, Q22 = 1.0345360431373387, Q23 = 1.6697050499054509, 
            Q33 = 1.0163160374692657, SoC_0 = 50.0):
    # initial state (SoC is intentionally set to a wrong value)
    # x = [[SoC], [RC voltage]]
    x = np.matrix([ [SoC_0],\
                    [0.0],\
                    [0.0]]
                    )

    exp_coeff1 = m.exp(-time_step / R1C1)
    exp_coeff2 = m.exp(-time_step / R2C2)
    # state transition model
    F = np.matrix([ [1, 0, 0         ],\
                    [0, exp_coeff1, 0],\
                    [0, 0, exp_coeff2]])

    # control-input model
    B = np.matrix([ [time_step / (Q_tot)],\
                    [ R1 * (1 - exp_coeff1)],\
                    [ R2 * (1 - exp_coeff2)]])

    R = R

    # state covariance
    P = np.matrix([ [R, 0, 0],\
                    [0, R, 0],\
                    [0, 0, R]])

    # process noise covariance matrix
    Q = np.matrix([ [Q11, Q12, Q13],\
                    [Q12, Q22, Q23],\
                    [Q13, Q23, Q33]])
    
    # Jacobian and Hx for charge
    def HJacobian_charge(x):
        return np.matrix([[ocp_charge_derivative(x[0, 0])*100, 1, 1]])

    def Hx_charge(x):
        return ocp_charge_fn(x[0, 0]) + x[1, 0] + x[2, 0]

    # Jacobian and Hx for discharge
    def HJacobian_discharge(x):
        return np.matrix([[ocp_discharge_derivative(x[0, 0])*100, 1, 1]])

    def Hx_discharge(x):
        return ocp_discharge_fn(x[0, 0]) + x[1, 0] + x[2, 0]

    return KalmanFilter(x, R0, F, B, P, Q, R, Hx_charge, Hx_discharge, HJacobian_charge, HJacobian_discharge)