import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
import random
import pickle

from utilities import *

from regression import *
import matplotlib
# matplotlib.use('TkAgg')

saveResultFile = True



'''
entry point of PRMCO, 
input_data:the search space bounds and dataset information (fixed and optimizable parameters for BF datasets, or the
dataset itself for map datasets).
'''
def find_extreme_in_blast(input_data, sampleSize=100, findMax=True, degree=3, iteration=3, generate_new_data=False,  seed=None, guess_size=1, method='COBYLA',  shrink_factor=2, epsilon=0.001, needPlotMap=True,error_mode = 0):
    input_array = input_data.input_array
    data = input_data.Z
    precisions = input_data.precisions
    fileName = input_data.fileName
    dimesion_info = input_data.dimesion_info
    origin_bounds = input_data.origin_bounds
    rmse_array = []
    r2_array = []
    mape_array = []
    adjusted_r2_array = []
    skip_reconstruct = True

    results = data
    bounds = origin_bounds
    
    real_extreme_point = input_data.real_extreme_point
    extreme_value = input_data.extreme_value 
    
    # For map datasets finds the true extreme point from the data set as a benchmark
    if generate_new_data == False and real_extreme_point is None:
        real_extreme_point, extreme_value = findRealExtreme(
            input_array, results, findMax)           
    print("real_extreme_point:", real_extreme_point)
    print("extreme_value", extreme_value)
    
    predict_points = []
    i = 0
    time_array = []

    converged = False
    needValData = True
    generate_sample_time = 0
    while i < iteration and not converged:
        print("bounds:", bounds)
        # Generate samples of BF data
        if generate_new_data is True:
            begin_time_gennerate_samples = time.time()
            sampled_values, sampled_results,val_values,val_results = generate_samples(
                fileName, sampleSize, bounds, dimesion_info, precisions, needValData=needValData)
            end_time_gennerate_samples = time.time()
            generate_sample_time = end_time_gennerate_samples - begin_time_gennerate_samples
        else:
            # Generate samples of map data
            sampled_values, sampled_results, val_values, val_results = getSamples(
                sampleSize, input_array, results, bounds, seed=seed, needValData=needValData)
            
        print("samplesize", len(sampled_values))
        # Because the resolution of map data is fixed, search space can not be shrinked indefinitely
        # When there is not enough samples, stop the algorithm
        low_bound_size = min(sampleSize, 1000)
        if len(sampled_values) < low_bound_size:
            print("not enough sample size ", len(sampled_values))
            break
        begin_time = time.time()
        # Train polynomial regression model
        model, poly, scaler = predict(
            sampled_values, sampled_results, degree, bounds)
        end_time = time.time()

        if len(val_results) > 0:
            rmse, r2, mape, adjusted_r2 = calculate_metrics(
                model, poly, scaler, val_values, val_results)
            rmse_array.append(rmse)
            r2_array.append(r2)
            mape_array.append(mape)
            adjusted_r2_array.append(adjusted_r2)
            print("rmse", rmse)

        if skip_reconstruct:
            print("skip reconstruct")
        else:
            coordinates, predicted_elevation = reconstruct_map(
                model, poly, scaler, bounds)

        # uses PSO (‘pyswarm’) or COBYLA (‘scipy’) to get the optimal input parameter combination 
        # within the search space, based on the objective function
        # provided by the trained model.
        extreme_points, extreme_values = findOptimum(
            model, poly, scaler, bounds, findMax, random_guess=guess_size, method=method)
        # Find the best solution from all parameter optimization experiments
        extreme_point = findExtremeInArray(
            extreme_points, extreme_values, findMax)
        # Visualization
        if needPlotMap and not skip_reconstruct:
            plotMap(scrollable_frame, predicted_elevation, coordinates,
                    extreme_points, real_extreme_point, extreme_point, title=i,origin_bounds=origin_bounds)

        print("find_from_predict_point", extreme_point, extreme_value)
        predict_points.append(extreme_point)
        # Calculate the reduced search space size 
        new_dimensions = getNewWH(
            origin_bounds, i, precisions, shrink_factor)
        # Contract the search space
        bounds = shrinkBound(origin_bounds, new_dimensions,
                             extreme_point, precisions=precisions)
        
        # Convergence determination
        if epsilon is not None:
            converged = lastNPointsConverged(
                epsilon, predict_points, origin_bounds, n=3)
        i = i+1
        time_array.append(end_time-begin_time+generate_sample_time)
    
    # Rescale the predicted input parameters to original scale    
    recovered_predict_points = []
    for predict_point in predict_points:
        recovered_predict_points.append(recoverPredictPoint(predict_point,input_data))
   
    if needPlotMap:
        plotMap(scrollable_frame, data, input_array,
                recovered_predict_points, real_extreme_point,origin_bounds=origin_bounds)
    # Calculate error
    error_array = calculateDistanceError(recovered_predict_points, real_extreme_point, input_data, origin_bounds, error_mode)
    return ExtremeResult(time_array, rmse_array, recovered_predict_points, real_extreme_point, r2=r2_array, mape_array=mape_array, adjusted_r2_array=adjusted_r2_array,error_array=error_array)

# Find maximum on mountain dataset
def findMap1():
    origin_data = np.loadtxt('data/taranaki_detail5120.txt')
    input_data = getElevationsMap(origin_data)
    result = find_extreme_in_blast(input_data, findMax=True,
                          sampleSize=500, degree=3, iteration=15, guess_size=10,shrink_factor=2)
    print(result)
# Find minimum on valley dataset
def findMap2():
    origin_data = np.loadtxt('data/minimum_detail.txt')
    input_data = getElevationsMap(origin_data)
    result = find_extreme_in_blast(input_data, findMax=False, sampleSize=5000,
                          degree=28, iteration=50, guess_size=100, method='COBYLA', shrink_factor=1.1, needPlotMap=True,epsilon=0.001)
    print(result)

# Find maximum on hills dataset
def findMap3():
    origin_data = np.loadtxt('data/palouse_detail5120.txt')
    input_data = getElevationsMap(origin_data)
    find_extreme_in_blast(input_data, findMax=True, sampleSize=50000, degree=9,
                          iteration=50,  guess_size=100, method='COBYLA', shrink_factor=1.1,epsilon=0.0002,error_mode=1)

# Find minimum on 3D BF dataset
def findBlastMap3D():
    input_data = getBlastMap3D()
    result = find_extreme_in_blast(input_data, findMax=False, sampleSize=560, degree=3, generate_new_data=True,
                          iteration=20,  guess_size=100, method='COBYLA', shrink_factor=1.1,  needPlotMap=True,epsilon=0.003)
    print(result)
# Find minimum on 2D BF dataset
def findBlastMap2D():
    input_data = getBlastMap()
    result = find_extreme_in_blast(input_data, findMax=False, sampleSize=210, degree=3, generate_new_data=True,
                          iteration=20,  guess_size=100, method='COBYLA', shrink_factor=1.1, needPlotMap=True,epsilon=0.003)
    print(result)
    
# Find optimum in 3D blast furnace dataset using pairwise method  
# Convert 3D full dimensional information to pairwise 2D information list
# Optimize iteratively
# combine result  
def findUsingPairwise():
    pairwise_inputs = generate_pairwise_inputs_3D()
    predict_points = []
    for index , input_data in enumerate(pairwise_inputs):
        if index != 0:
            input_old = pairwise_inputs[index-1]
            # Learn parameter from last iteration
            setOptimisedConstant(input_old,input_data,predict_points[-1])
        result = find_extreme_in_blast(input_data, findMax=False, sampleSize=210, degree=3, generate_new_data=True,
                          iteration=20,  guess_size=100, method='COBYLA', shrink_factor=1.1, needPlotMap=False,epsilon=0.003)
        predict_points = result.predict_points

    print("combined", getCombinedResult(pairwise_inputs[-1],predict_points[-1]))
    
scrollable_frame = None

blast_fileName2 = "blast_furnace_results2.csv"
 

def real_run():
    unzip()
    # The following five methods can all be run. 
    # Uncomment the desired method to execute it.
    
    # Find maximum on mountain dataset
    findMap1()
    
    # Find minimum on valley dataset
    # findMap2()
    
    # Find maximum on hills dataset
    # findMap3()
    
    # Find minimum on 2D BF dataset
    # findBlastMap2D()
    
    # Find minimum on 3D BF dataset
    # findBlastMap3D()


def main():
    root = tk.Tk()
    root.title("Scrollable Matplotlib Figures")
    root.geometry("1000x800")
    
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient='vertical', command=canvas.yview)
    global scrollable_frame
    scrollable_frame = ttk.Frame(canvas)

   
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    real_run()

    
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    root.mainloop()


if __name__ == "__main__":
    main()
