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
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
from utilities import *
from regression import *
# 0 mean, else extreme
default_extreme_mode = 0
saveResultFile = True

# performs clustering
def divide_to_blocks(bounds, block_nums, precisions):
    num_dimensions = len(bounds)
    assert num_dimensions == len(block_nums) == len(precisions), "wrong"

    divisions = [np.linspace(bounds[i][0], bounds[i][1], block_nums[i] + 1) for i in range(num_dimensions)]
    bounds_array = []

    for index in np.ndindex(*[len(div)-1 for div in divisions]):
        current_bounds = [(divisions[i][index[i]], divisions[i][index[i] + 1]) for i in range(num_dimensions)]
        rounded_bounds = [(round(bound[0], precisions[i]), round(bound[1], precisions[i])) for i, bound in enumerate(current_bounds)]
        bounds_array.append(rounded_bounds)
    return bounds_array,divisions

# get samples in each cluster and identifies the optimal cluster using either the extreme value mode or the average mode
def findExtremeBlock(input_array, data, bounds_array, sample_size, findMax=False, need_generate_data=False, fileName=None, dimesion_info=None, precisions=None,mode = default_extreme_mode,extreme_sample_num = 3):
    best_mean = None
    best_block = None
    
    # get samples
    if need_generate_data:
        split_points, split_values = generate_samples_all(fileName, bounds_array, sample_size, dimesion_info, precisions)
    else:
        split_points = []
        split_values = []
        for bound in bounds_array:
            train_values, train_results, _, _ = getSamples(sample_size, input_array, data, bound,outer_sample_factor = 0)
            split_points.append(train_values)
            split_values.append(train_results)

    # identifies the optimal cluster using either the extreme value mode or the average mode
    for i, bound in enumerate(bounds_array):
  
        train_results = split_values[i]
        if len(train_results) <= 0:
            return best_block, best_mean,len(train_results)
        
        if mode == 0:
            evaluate_value = np.mean(train_results)
        else:
            if findMax:
                top_3_max = np.sort(train_results)[-extreme_sample_num:]  
                evaluate_value = np.mean(top_3_max)  
            else:
                bottom_3_min = np.sort(train_results)[:extreme_sample_num]  
                evaluate_value = np.mean(bottom_3_min)  
        
        if best_mean is None or (findMax and evaluate_value > best_mean) or (not findMax and evaluate_value < best_mean):
            best_mean = evaluate_value
            best_block = bound
    return best_block, best_mean,len(split_values[0])

'''
entry point of CMCO, 
input_data:the search space bounds and dataset information (fixed and optimizable parameters for BF datasets, or the
dataset itself for map datasets).
'''
def find_extreme_use_cluster(input_data,block_nums, sampleSize=100, findMax=True, iteration=3, generate_new_data=False, seed=None, shrink_factor=2,needPlotMap = False,epsilon = 0.003,error_mode = 0):
    input_array = input_data.input_array
    data = input_data.Z
    precisions = input_data.precisions
    fileName = input_data.fileName
    dimesion_info = input_data.dimesion_info
    
    origin_bounds = input_data.origin_bounds  
    bounds = origin_bounds
    
    real_extreme_point = input_data.real_extreme_point
    extreme_value = input_data.extreme_value 
    # For map datasets finds the true extreme point from the data set as a benchmark
    if generate_new_data == False and real_extreme_point is None:
        real_extreme_point, extreme_value = findRealExtreme(
        input_array, data, findMax)
        
    print("real_extreme_point:", real_extreme_point)
    print("extreme_value", extreme_value)
    predict_points = []
    i = 0
  
    time_array = []
    converged = False
    while i < iteration and not converged:
        # performs clustering
        bounds_array,divisions = divide_to_blocks(
            bounds, block_nums,precisions)
        print("bounds:",bounds)
        begin_time = time.time()
        # get samples in each cluster and identifies the optimal cluster using either the extreme value mode or the average mode
        best_block, best_mean,real_sample_size = findExtremeBlock(
            input_array, data, bounds_array, sampleSize, findMax=findMax,need_generate_data = generate_new_data,fileName=fileName,dimesion_info=dimesion_info,precisions = precisions)
        print("real_sample_size:",real_sample_size)
        if real_sample_size <= 0:
        # if real_sample_size < sampleSize:
            print("not enough samplesize")
            break
        time_array.append(time.time()-begin_time)
        # returns the centroid of the optimal cluster as the predicted optimal input parameter combination
        extreme_point = returnPointInBlock(best_block, data,precisions)
        print("find_from_predict_point", extreme_point)
        predict_points.append(extreme_point)
        # calculate the reduced search space size 
        new_dimensions = getNewWH(
            origin_bounds, i, precisions, shrink_factor)
        # contract the search space
        bounds = shrinkBound(origin_bounds, new_dimensions,
                             extreme_point, precisions=precisions)
        # convergence determination
        if epsilon is not None:
            converged = lastNPointsConverged(epsilon,predict_points,origin_bounds, n=3)
            if converged:
                print("converged",block_nums)
        i = i+1
        
    
    if needPlotMap:
        plotMap(scrollable_frame, data, input_array,
            predict_points, real_extreme_point,origin_bounds=origin_bounds)
    error_array = calculateDistanceError(predict_points, real_extreme_point, input_data, origin_bounds, error_mode)
    return ExtremeResult(time_array,  0, predict_points, real_extreme_point,error_array=error_array)


def returnPointInBlock(best_block_bounds, data, precisions):
    num_dimensions = len(best_block_bounds)
    mid_points = []

    for dim in range(num_dimensions):
        min_bound, max_bound = best_block_bounds[dim]
        precision = precisions[dim]
        mid_point = round((min_bound + max_bound) / 2, precision)
        mid_points.append(mid_point)

    return mid_points


def plotMap1():
    origin_data = np.loadtxt('data/taranaki_detail5120.txt')
    input_data = getElevationsMap(origin_data)
    block_nums = [2,2]
    result = find_extreme_use_cluster(input_data, block_nums,findMax=True, sampleSize=100,
                                    iteration=50,  generate_new_data=False,  shrink_factor=2,needPlotMap = True)   
    print(result)
    return result
def plotMap2():
    origin_data = np.loadtxt('data/minimum_detail5120.txt')
    input_data = getElevationsMap(origin_data)
    block_nums = [23,23]
    result = find_extreme_use_cluster(input_data,  block_nums,findMax=False, sampleSize=100, 
                                    iteration=50,  generate_new_data=False, shrink_factor=1.2,needPlotMap = True)
    print(result)
    return result

def plotMap3():
    origin_data = np.loadtxt('data/palouse_detail5120.txt')
    input_data = getElevationsMap(origin_data)
    block_nums = [10,10]
    result = find_extreme_use_cluster(input_data, block_nums,findMax=True, sampleSize=100, 
                                    iteration=50,  generate_new_data=False,  shrink_factor=1.2,needPlotMap = True,error_mode=1,epsilon=0.0002)
    print(result)
    return result
def findBlastMap2D(): 
    block_nums = [2,2]
    input_data = getBlastMap()   
    result = find_extreme_use_cluster(input_data, block_nums,findMax=False, sampleSize=16, 
                                    iteration=30,  generate_new_data=True,  shrink_factor=2,needPlotMap= True)
    print(result)
def findBlastMap3D():
  
    
    block_nums = [2,2,2]
    input_data = getBlastMap3D(isCluster=True)  

    result =   find_extreme_use_cluster(input_data, block_nums,findMax=False, sampleSize=64, 
                                    iteration=30,  generate_new_data=True,  shrink_factor=2,needPlotMap= True)
   
    print(result)

scrollable_frame = None


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



def real_run():
    unzip()
    # The following five methods can all be run. 
    # Uncomment the desired method to execute it.
    
    # find maximum on mountain dataset
    plotMap1()
    
    # find minimum on valley dataset
    # plotMap2()
    
    # find maximum on hills dataset
    # plotMap3()
    
    # find minimum on 2D BF dataset
    # findBlastMap2D()
    
    # find minimum on 3D BF dataset
    # findBlastMap3D()
  

if __name__ == "__main__":
    main()

