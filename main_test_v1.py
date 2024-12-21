
# This code was written by the Huawei Technologies Düsseldorf GmbH team for the TechArena evaluation process.
# The purpose of this code is to calculate the Root Mean Square Error (RMSE), 
# the Maximum Absolute Error (MAE), and the running time of the algorithm named 'soc_prediction'.

# Instructions:
    
# Attached Code: You will find a Python code in the attachment. This code is provided as an example to help you test and evaluate your own algorithm.

# Real SOC Values: Keep in mind that the real SOC values are not actually known in real-world scenarios, but they are given here to help you assess the accuracy of your code.

# Initial Error: To make the testing more realistic, an initial error must be considered. The first SOC value is typically known in real situations, but it can also have an inherent error.
# For example, if the actual initial SOC is 50%, it could range from 35% to 65%, with a potential error of �15%.

# Testing Consideration: While evaluating your algorithm, assume that the initial SOC error could increase or decrease by 15%. This range will help you assess how well your algorithm handles initial inaccuracies.

# Algorithm Convergence: It’s important to note that most algorithms will need some time to converge and minimize the error between the real and estimated SOC values.

# RMSE and MaxAE Calculation: To help you further, the RMSE and MaxAE will be calculated only after 3 days of running data (relax_time = 259200 samples) to give the algorithm time to stabilize.




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from model_test import soc_prediction
from Kalman_filter import get_EKF
import time

def error_function(arrymena, arrystd, dfsum):
    return np.random.normal(loc=arrymena, scale=arrystd, size=dfsum)

file_name = 'GenerateTestData_R6S2B1_DAY0to4-injectionTemplate.xlsx'
df = pd.read_excel(file_name, skiprows=7, names=['Step', 'voltage', 'current', 'SOC_true', 'SOC_Calc', 'Error', 'temperature','AS','Out_current','Comment'])
#%%
# Input signal noise
sumcount = df.shape[0]
df['current'] = df['current'] + error_function(2, 1, sumcount)
df['voltage'] = df['voltage'] + error_function(3 * 0.001, 1 * 0.001, sumcount)
df['temperature'] = df['temperature'] + error_function(3, 1, sumcount)

def read_excel():
    voltage = df['voltage'].tolist()
    current = df['current'].tolist()
    temperature = df['temperature'].tolist()
    soc_true = df['SOC_true'].tolist()
    return voltage, current, temperature, soc_true

def calculate_rmse(true_values, predicted_values):
    """ Calculate Root Mean Squared Error (RMSE) """
    return np.sqrt(np.mean((np.array(predicted_values) - np.array(true_values)) ** 2))

def calculate_mae(true_values, predicted_values):

    """ Calculate MaxAE Absolute Error (MaxAE) """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    return np.max(np.abs(true_values - predicted_values))
#%%
def main():
    
    voltage, current, temperature, real_soc = read_excel()
    predicted_soc = [] 
    Relaxed_predicted_soc = []
    Initial_soc = 60
    relax_time = 259200  # if There is no intial error, relax time is 0

    # Kalman filter inisialization
    kf = get_EKF(SoC_0=Initial_soc)
    
    # Measure the start time
    start_time = time.time()

    for i in range(len(current)):
        current_1 = current[i]
        voltage_1 = voltage[i]
        
        # My code starts
        # Kalman Filter steps
        kf.predict(u=current_1)  # Predict next state
        kf.update(voltage_1, u=current_1)  # Update state with measurement 
        predicted_soc.append(kf.x[0, 0])
        # My code ends

        if i >= relax_time :  
            Relaxed_predicted_soc.append(kf.x[0, 0])
            
        # print(f"Sample {i+1}: Voltage = {voltage[i]}V, Current = {current[i]:.4f}A, Temperature = {temperature[i]:.4f}C, Predicted SOC = {current_soc:.2f}%, True SOC = {real_soc[i]:.2f}%")
       
    # Measure the end time
    end_time = time.time()
    
    
    # Calculate the total execution time
    total_time = end_time - start_time

    # Calculate RMSE and MAE
    rmse = calculate_rmse(real_soc[relax_time:], Relaxed_predicted_soc)
    mae = calculate_mae(real_soc[relax_time:], Relaxed_predicted_soc)

    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Total Execution Time: {total_time:.4f} seconds")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(real_soc, label='Real SOC', color='blue', linestyle='-', marker='o', markersize=1)
    plt.plot(predicted_soc, label='Predicted SOC', color='red', linestyle='--', marker='x', markersize=1)
    plt.axvline(x=relax_time, color='black', linestyle=':', linewidth=3)  # Discontinuous line

    plt.text(0.67, 0.20, f'RMSE: {rmse:.4f}', transform=plt.gca().transAxes, fontsize=12,verticalalignment='top', horizontalalignment='left', color='black')
    plt.text(0.67, 0.15, f'MaxAE: {mae:.4f}', transform=plt.gca().transAxes, fontsize=12,verticalalignment='top', horizontalalignment='left', color='black')
    plt.text(0.67, 0.10, f'Total Execution Time: {total_time:.4f} s', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left', color='black')
 
    
    plt.title('Real vs Predicted SOC')
    plt.xlabel('Sample Index')
    plt.ylabel('SOC (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

#%%

if __name__ == "__main__":
    main()