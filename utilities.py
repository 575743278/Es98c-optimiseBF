import matplotlib.pyplot as plt
import numpy as np
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
from dataclasses import dataclass, field
import pandas as pd
import os
from metlab_api import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
from regression import *
import zipfile
import itertools
import numpy as np

savePlot = True
plot_dic = 'result_plot1'
map_name ='try'
number_file_path = './result_file/number.txt'
@dataclass
class ExtremeResult:
    time_array: np.ndarray
    rmse: np.ndarray
    predict_points: np.ndarray = None
    real_extreme_point: np.ndarray = None
    error_array: np.ndarray = None
    r2: np.ndarray = None
    mape_array :np.ndarray = None
    adjusted_r2_array:np.ndarray = None

# @dataclass
# class OneDegreeResult:
#     extreme_result_array: list


@dataclass
class DimensionInfo:
    variables_name: list
    constants_map: dict
    factors: dict = field(default_factory=dict)
    desired_order: np.ndarray = field(default_factory=lambda: np.array(['CokeOreRatio', 'HotBlastRate', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']))
    need_log_column: np.ndarray = field(default_factory=lambda: np.array(['CO2_Fe_Ratio']))

@dataclass
class InputData:
    input_array: np.ndarray
    Z: np.ndarray
    precisions: np.ndarray
    fileName: str = ""
    dimesion_info: DimensionInfo = None
    map_name = ""
    origin_bounds : np.ndarray = None
    real_extreme_point : np.ndarray = None
    extreme_value : float = None
    factor_array : np.ndarray = None
    
    
    
@dataclass
class MapInput:
    x_lable: str = None
    y_lable: str = None
    z_lable: str = None
    elevation:str = None
    

            
variables_list = ['CokeOreRatio', 'HotBlastRate',
                  'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']

# Combine generated input parameters with other fixed input parameters to format for BF model acceptance.
def convert_points_2_inputparams(samplePoints, dimensionInfo):
    variables_name = dimensionInfo.variables_name
    constants_map = dimensionInfo.constants_map

    all_variable_names = variables_name + list(constants_map.keys())
    
    file_path = '/Users/han/mymap/input_file.csv'
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(all_variable_names)

        for point in samplePoints:
            input_params = list(point) + [constants_map[key]
                                          for key in constants_map]
            writer.writerow(input_params)

    print("CSV file 'input_file.csv' has been updated.")
    
# Reorder the input parameters to blast furnace model needed format
def reorder_columns(file_path, output_path,dimension_info):
    
    df = pd.read_csv(file_path)
    
    desired_order = dimension_info.desired_order
    
    df = df[desired_order]
    
    df.to_csv(output_path, index=False)


# Split the parameter set into multiple files for parallel processing by the blast furnace model.
def divideData1():
    input_file = '/Users/han/mymap/input_file.csv'
    data = pd.read_csv(input_file)

    num_rows = len(data)
    rows_per_file = num_rows // 10

    for i in range(10):
        start_row = i * rows_per_file
        if i == 9: 
            end_row = num_rows
        else:
            end_row = (i + 1) * rows_per_file

        subset = data.iloc[start_row:end_row]

        output_file = f'/Users/han/mymap/input_file3_{i + 1}.csv'

        subset.to_csv(output_file, index=False)
# Log transformation
def log_transform_columns(input_file_path, output_file_path, dimension_info):
    df = pd.read_csv(input_file_path)
    
    need_log_columns = dimension_info.need_log_column
    
    for column in need_log_columns:
        if column in df.columns:
            df[column] = np.log1p(df[column])
        else:
            print(f"Column '{column}' not found in {input_file_path}")
    
    df.to_csv(output_file_path, index=False)
    
    print(f'Modified file saved to {output_file_path}')    
    
# Scale or restore the original scale if the search range has been scaled.
def multiply_columns_by_factors(input_file_path, output_file_path, dimensionInfo,recover = True):
    df = pd.read_csv(input_file_path)
    
    factors = dimensionInfo.factors
    if factors is None:
        return
    for column, factor in factors.items():
        if column in df.columns:
            if recover:
                df[column] = df[column] * factor[1]
            else: 
                df[column] = df[column] * factor[0]
        else:
            print(f"Column '{column}' not found in {input_file_path}")
    
    df.to_csv(output_file_path, index=False)
    
    print(f'Modified file saved to {output_file_path}')
      
# Merge the generated samples.
def combineData1(file_path_pattern,output_file_path):
    combined_df = pd.DataFrame()

    for i in range(1, 11):
        file_path = file_path_pattern.format(i)
        
        df = pd.read_csv(file_path)
        
        if i == 1:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file_path, index=False)

    print(f'Combined file saved to {output_file_path}')
    
# Extract samples from files to array
def extract_points_from_csv(csv_filename, dimensionInfo):
    variables_name = dimensionInfo.variables_name

    points = []
    ratios = []

    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            point = [float(row[var]) for var in variables_name]
            points.append(point)
            ratios.append(float(row['CO2_Fe_Ratio']))

    return np.array(points), np.array(ratios)

# Visualization 
def plotMap(frame, data, coordinates, predictPoints=None, readExtreme=None, bestPredictPoints=None, bounds=None, divisions=None, title='',mapInput =  MapInput('','','',''),origin_bounds = None):
    if len(origin_bounds) == 2:
        plotMap2D(frame, data, coordinates, predictPoints, readExtreme,
                  bestPredictPoints, bounds, divisions, title,mapInput)
    elif len(origin_bounds) == 3:
        plotMap3D(frame, data, coordinates, predictPoints, readExtreme,
                  bestPredictPoints, bounds,  title,mapInput)

number = 1
# Visualization in 2D
def plotMap2D(frame, data, coordinates, predictPoints=None, readExtreme=None, bestPredictPoints=None, bounds=None, divisions=None, title='',mapInput = None):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel(mapInput.x_lable)
    ax.set_ylabel(mapInput.y_lable)
    if data is not None and len(data)>0:
        contour = ax.contourf(
            coordinates[1], coordinates[0], data, 20, cmap='viridis')
        fig.colorbar(contour, ax=ax, label=mapInput.elevation)
    if predictPoints is not None:
        num_points = len(predictPoints)
        cmap = plt.get_cmap('cool')
        for idx, points in enumerate(predictPoints):
            label = 'Predict Points' if idx == 0 else "" 
            ax.plot(points[1], points[0], 'o', color=cmap(
                idx / num_points), label=label, markersize=4)
    if readExtreme is not None:
        ax.plot(readExtreme[1], readExtreme[0], 'rx', label='Extreme Point')
    if bestPredictPoints is not None:
        ax.plot(bestPredictPoints[1], bestPredictPoints[0],
                'yo', label='bestPredictPoints')
    
    ax.legend(loc='lower left')
    # ax.legend(loc='upper right')
    if bounds is not None:
        min_lat, max_lat = bounds[0]
        min_lon, max_lon = bounds[1]
        ax.plot([min_lon, max_lon, max_lon, min_lon, min_lon],
                [min_lat, min_lat, max_lat, max_lat, min_lat], 'r-', linewidth=2, label='Bounds')
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    global number
    number = number+1
    # save_plot(fig,number)
    # plt.close(fig)


# Visualization in 3D
def plotMap3D(frame, data, coordinates, predictPoints=None, readExtreme=None, bestPredictPoints=None, bounds=None, title='',mapInput = None):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(mapInput.x_lable)
    ax.set_ylabel(mapInput.y_lable)
    ax.set_zlabel(mapInput.z_lable)
    ax.set_zlabel(mapInput.z_lable, labelpad=-1)
  
    if predictPoints is not None:
        num_points = len(predictPoints)
        cmap = plt.get_cmap('cool')
        for idx, points in enumerate(predictPoints):
            label = 'Predict Points' if idx == 0 else "" 
            ax.scatter(points[0], points[1], points[2], color=cmap(
                idx / num_points), label=label, s=20)

    if readExtreme is not None:
        ax.scatter(readExtreme[0], readExtreme[1], readExtreme[2],
                   color='red', marker='x', label='Extreme Point')
        
    if bestPredictPoints is not None:
        ax.scatter(bestPredictPoints[0], bestPredictPoints[1], bestPredictPoints[2],
                   color='yellow', marker='o', label='Best Predict Points')

    
    ax.legend(loc='upper left')
    plt.tight_layout()
 
    # save_plot(fig,title)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()


# Get samples from existed dataset (Map datasets)
def getSamples(sampleSize, input_array, results, bounds, seed=None, train_ratio=0.8, needValData=False,outer_sample_factor = 0.2,pure_no_border = False):
    np.random.seed(seed)
    num_outer_samples =int(outer_sample_factor*sampleSize) 
    originSampleSize = sampleSize
    if needValData:
        sampleSize = int(sampleSize / train_ratio) + 5

    num_dimensions = len(input_array)
    print("len(bounds)",len(bounds))
    print(len(input_array))
    assert num_dimensions == len(bounds), "wrong"

    valid_ranges = []
    for i in range(num_dimensions):
        min_bound, max_bound = bounds[i]
        valid_range = np.where((input_array[i] >= min_bound) & (
            input_array[i] <= max_bound))[0]
        valid_ranges.append(valid_range)

   
    cropped_results = results[np.ix_(*valid_ranges)]

    
   
    valid_indices = np.meshgrid(*valid_ranges, indexing='ij')
    sampled_values = np.array(
        [input_array[i][valid_indices[i]].flatten() for i in range(num_dimensions)]).T
    sampled_results = cropped_results.flatten()
    
    if num_outer_samples > 0:
        
        reshaped_sampled_values = sampled_values.reshape(cropped_results.shape + (num_dimensions,))
        outer_values = extract_outer_layer(reshaped_sampled_values, mode='ignore_last')
        outer_results = extract_outer_layer(cropped_results)
        print("outer sample:",len(outer_values))
        print("sample size:",len(sampled_values))
        
   
        if len(outer_values) < num_outer_samples:
            num_outer_samples = len(outer_values)
        outer_sample_indices = np.random.choice(len(outer_values), size=num_outer_samples, replace=False)
        outer_values = outer_values[outer_sample_indices]
        outer_results = outer_results[outer_sample_indices]

   
    if len(sampled_values) < sampleSize:
        sampleSize = len(sampled_values)
    # print(sampled_values)

  
    sampled_indices = np.random.choice(len(sampled_values), size=sampleSize, replace=False)
    sampled_values = sampled_values[sampled_indices]
    sampled_results = sampled_results[sampled_indices]
    
    if num_outer_samples > 0:

       
        values_with_border = np.concatenate((sampled_values, outer_values), axis=0)
        results_with_border = np.concatenate((sampled_results, outer_results), axis=0)
        print("sampled_values",len(sampled_values))
        print("outer_values",len(outer_values))
        
        print("values_with_border1",len(outer_values)+len(sampled_values))
        
   
        unique_indices = np.unique(values_with_border, axis=0, return_index=True)[1]
        values_with_border = values_with_border[unique_indices]
        results_with_border = results_with_border[unique_indices]
        print("values_with_border1",len(values_with_border))

    else:
        values_with_border = sampled_values
        results_with_border = sampled_results

    if not needValData:
        return values_with_border, results_with_border, [], []
    return seperateTrainVal(values_with_border, results_with_border,train_ratio,originSampleSize)
    

    
# Seperate train and validate data
def seperateTrainVal(values_with_border,results_with_border,train_ratio,originSampleSize):
  
    total_samples = len(values_with_border)
    train_size = int(total_samples * train_ratio)

    if(train_size < originSampleSize):
        train_size = originSampleSize
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_values = values_with_border[train_indices]
    train_results = results_with_border[train_indices]
    val_values = values_with_border[val_indices]
    val_results = results_with_border[val_indices]
    return train_values, train_results, val_values, val_results

# Sample input parameter combinations within the search range.
def generateRandomPoints(sampleSize, bounds,  precisions=None, mode=1, outer_sample_factor = 0.01):
    num_dimensions = len(bounds)
    
    if precisions is None:
        precisions = [10] * num_dimensions 
    assert len(bounds) == len(precisions), "wrong"

    if mode == 1:
        
        samples = []
        for i in range(num_dimensions):
            min_bound, max_bound = bounds[i]
            samples.append(np.random.uniform(min_bound, max_bound, sampleSize))
        
    else:
        raise ValueError(
            "Unsupported mode. Only mode 1 (random) is supported.")
    samples = np.column_stack(samples)

    for i in range(num_dimensions):
        samples[:, i] = np.round(samples[:, i], precisions[i])
    
    outer_samples_num = int(sampleSize * outer_sample_factor)
    if outer_samples_num > 0:
        boundary_samples = boundary_sampling(bounds, outer_samples_num)
            
            
        samples = np.vstack([samples, boundary_samples])
    return samples

# Generate input parameter samples on boundaries
def boundary_sampling(bounds, total_boundary_points):

    num_dimensions = len(bounds)
    
  
    samples_per_edge = total_boundary_points // (2 * num_dimensions)
    
    samples = np.empty((0, num_dimensions))

  
    for dim in range(num_dimensions):
        
        dim_samples = np.random.rand(2 * samples_per_edge, num_dimensions)
        
        min_val, max_val = bounds[dim]
        dim_samples[:, dim] = np.concatenate([np.full(samples_per_edge, min_val),
                                              np.full(samples_per_edge, max_val)])
        
     
        for other_dim in range(num_dimensions):
            if other_dim != dim:
                dim_samples[:, other_dim] = np.random.uniform(bounds[other_dim][0], bounds[other_dim][1], 2 * samples_per_edge)
        
       
        samples = np.vstack([samples, dim_samples])

    return samples


'''
For PRMCO.
1. Sample input parameter combinations within the search range.
2. Combine these with other fixed input parameters to format for BF model acceptance.
3. Restore the original scale if the search range has been scaled.
4. Split the parameter set into multiple files for parallel processing by the blast furnace model.
5. Merge the generated samples.
6. Apply a logarithmic transformation.
'''

def generate_samples(saveFileName, sampleSize, bounds, dimension_info, precisions, mode=1,outer_sample_factor = 0.2, train_ratio=0.8, needValData=False):  
    originSampleSize = sampleSize
    needValData = True
    if needValData:
        print("needValData",needValData)
        sampleSize = int(sampleSize / train_ratio) + 5   
    points = generateRandomPoints(sampleSize, bounds, precisions, mode,outer_sample_factor)
    convert_points_2_inputparams(points, dimension_info)
    input_file_name= '/Users/han/mymap/input_file.csv'
    reorder_columns(input_file_name,input_file_name,dimension_info)
    multiply_columns_by_factors(input_file_name,input_file_name,dimension_info)
    divideData1()
    simulation_generate_results(saveFileName)
    file_path_pattern = '/Users/han/mymap/blast_results_from_file3_{}.csv'  
    combineData1(file_path_pattern,saveFileName)
    multiply_columns_by_factors(saveFileName,saveFileName,dimension_info,recover=False)
    log_transform_columns(saveFileName,saveFileName,dimension_info)
    # backup_data(saveFileName, 'back_up.csv')
    existed_points, existed_values = extract_points_from_csv(
        saveFileName, dimension_info)

    if not needValData:
      return existed_points, existed_values,[],[]
    print("seperateTrainVal",seperateTrainVal)
    return seperateTrainVal(existed_points, existed_values,train_ratio,originSampleSize)


'''
For CMCO.
This method combines the samples from all clusters in one iteration for a single simulation, 
and then the results are divided and returned to each respective cluster to save time.
1. Sample input parameter combinations within the search range.
2. Combine these with other fixed input parameters to format for BF model acceptance.
3. Restore the original scale if the search range has been scaled.
4. Split the parameter set into multiple files for parallel processing by the blast furnace model.
5. Merge the generated samples.
6. Apply a logarithmic transformation.
'''
def generate_samples_all(saveFileName, all_bounds, sampleSize, dimension_info, precisions, mode=1):
    all_points = []
    bound_indices = []  

    for bounds in all_bounds:
        points = generateRandomPoints(sampleSize, bounds, precisions, mode,0)
        all_points.extend(points)
        bound_indices.append((len(all_points) - len(points), len(all_points)))
    print("bound_indices:", bound_indices)
    all_points = np.vstack(all_points)  
    convert_points_2_inputparams(all_points, dimension_info)
    input_file_name= '/Users/han/mymap/input_file.csv'
    reorder_columns(input_file_name,input_file_name,dimension_info)
    multiply_columns_by_factors(input_file_name,input_file_name,dimension_info)
    divideData1()
    simulation_generate_results(saveFileName)
    file_path_pattern = '/Users/han/mymap/blast_results_from_file3_{}.csv'  
    combineData1(file_path_pattern,saveFileName)
    multiply_columns_by_factors(saveFileName,saveFileName,dimension_info,recover=False)
    log_transform_columns(saveFileName,saveFileName,dimension_info)

    existed_points, existed_values = extract_points_from_csv(
        saveFileName, dimension_info)

    split_points = []
    split_values = []

    for start, end in bound_indices:
        split_points.append(existed_points[start:end])
        split_values.append(existed_values[start:end])
    print("generate_samples_all")
    print("split_points", split_points)
    print("split_values", split_values)
    print("bounds_array", all_bounds)
    return split_points, split_values

#2D Map dataset information
def getBlastMap(plotContour = False):
    input_array = None
    Z = None
    precisions = [10, 10]
    
    blast_fileName_MD = "bf_MD.csv"
    variables_name = ['CokeOreRatio', 'HotBlastRate']
    constants_map = {"f1": 0.0000001,
                     "f2": 0.0000001,
                     "f3": 0.0000001,
                     "f4": 0.0000001,
                     "f5": 0.0000001,
                     "f6": 0.0000001,
                     "f7": 0.0000001}
    info = DimensionInfo(variables_name, constants_map)
    
    origin_bounds =[(1,9),(10,150)]
    return InputData(input_array, Z, precisions,blast_fileName_MD,info,origin_bounds=origin_bounds,real_extreme_point=(9,150))



def getMultiplyFactor(variables_name, factors):

    factor_array = []
    
    for variable in variables_name:
        if variable in factors:
            factor_array.append(factors[variable])
        else:
           
            factor_array.append([1,1])
    
    return factor_array


# Scale search space    
def getBoundsMultiplied(bounds, factor_array, precision_array):
   
    multiplied_bounds = []
    
    for bound, factor, precision in zip(bounds, factor_array, precision_array):
        min_bound, max_bound = bound
        new_min_bound = round(min_bound * factor[0], precision)
        new_max_bound = round(max_bound * factor[0], precision)
        new_bound = (new_min_bound, new_max_bound)
        multiplied_bounds.append(new_bound)
    
    return multiplied_bounds

# Predicted input parameter has been scaled,need to rescale
def recoverPredictPoint(predict_point, input_data):
  
    factor_array = input_data.factor_array
    precision_array = input_data.precisions
    if factor_array is None :
        return predict_point
    recovered_point = []
    for point, factor, precision in zip(predict_point, factor_array, precision_array):
        recovered_value = round(point * factor[1], precision)
        recovered_point.append(recovered_value)
    
    return recovered_point


# 3D visulization
def getBlastMap3D(fileName = None,isCluster = False):
    input_array = None
    Z = None
    
    precisions = [10, 10, 15]
    
    variables_name = ['CokeOreRatio', 'HotBlastRate', 'f1']
    constants_map = {"f2": 0.0000001,
                     "f3": 0.0000001,
                     "f4": 0.0000001,
                     "f5": 0.0000001,
                     "f6": 0.0000001,
                     "f7": 0.0000001}
    info = DimensionInfo(variables_name, constants_map)
    blast_fileName_MD = "bf_MD.csv"
    
    origin_bounds =[(1,9),(10,150),(2e-8,1.9e-7)]
    factor_array = None
    # Due to precision issues with the built-in optimization algorithms in the range (2e-8, 1.9e-7),
    # the range is scaled to (0.2, 1.9) for parameter search. The original magnitude is restored
    # when passing the input parameters to the blast furnace model.
    if isCluster == False:
        info.factors = { "f1": [1e7,1e-7]}
        factor_array = getMultiplyFactor(variables_name, info.factors)
        origin_bounds = getBoundsMultiplied(origin_bounds, factor_array,precisions)
    return InputData(input_array, Z, precisions,blast_fileName_MD,info,origin_bounds=origin_bounds,real_extreme_point=(9,150,1.9e-7),factor_array=factor_array)
    

# Find the true optimum in map datasets as benchmark
def findRealExtreme(input_array, results, findMax=True):
  
    if findMax:
        extreme_value = -np.inf
    else:
        extreme_value = np.inf
    extreme_index = None

 
    for index, value in np.ndenumerate(results):
        if findMax:
            if value > extreme_value:
                extreme_value = value
                extreme_index = index
        else:
            if value < extreme_value:
                extreme_value = value
                extreme_index = index

    extreme_point = [input_array[i][idx]
                     for i, idx in enumerate(extreme_index)]

    return extreme_point, extreme_value

# Contract the search space
def shrinkBound(origin_bounds, new_sizes, center_point, precisions=None):
    num_dimensions = len(origin_bounds)
    assert num_dimensions == len(new_sizes) == len(
        center_point), "wrong"

    new_bounds = []

    for i in range(num_dimensions):
        min_bound, max_bound = origin_bounds[i]
        new_size = new_sizes[i]
        center = center_point[i]
        precision = precisions[i]

       
        new_min_bound = max(min_bound, center - new_size / 2)
        new_max_bound = min(max_bound, center + new_size / 2)

        if new_min_bound == min_bound:
            new_max_bound = min(max_bound, new_min_bound + new_size)
        elif new_max_bound == max_bound:
            new_min_bound = max(min_bound, new_max_bound - new_size)

        new_bounds.append((round(new_min_bound, precision),
                          round(new_max_bound, precision)))

    
    return new_bounds

# Calculate the length of each dimension of the reduced search space  
def getNewWH(origin_bounds, iteration, precisions=0, shrink_factor=2):
    iteration = iteration + 1
    divide = shrink_factor ** iteration
    print("iteration:", iteration)
    print("divide:", divide)

    new_dimensions = []
    for i, bound in enumerate(origin_bounds):
        min_bound, max_bound = bound
        size = (max_bound - min_bound) / divide
        rounded_size = round(size, precisions[i])
        new_dimensions.append(rounded_size)

    return new_dimensions


# Find the optimal solution by identifying the extremum obtained from multiple parameter optimization tests.
def findExtremeInArray(extreme_points, extreme_values, findMax=False):
    index = 0
    if findMax:
        index = np.argmax(extreme_values)
    else:
        index = np.argmin(extreme_values)
    extreme_point = extreme_points[index]
    return extreme_point

# For plot contour
def getElevationsMap(origin_data, bounds=None):
    lats = np.arange(origin_data.shape[0])
    lons = np.arange(origin_data.shape[1])
    input_array = []
    input_array.append(lats)
    input_array.append(lons)
    precisions = [0, 0]
    origin_bounds = []
    for inputs in input_array:
        origin_bounds.append((inputs[0], inputs[-1]))
    return InputData(input_array, origin_data, precisions,origin_bounds=origin_bounds)


# Normalize predicted input parameters to calculate the relative parameter error not the absolute error
def normalize_point(point, bounds):
   
    bounds = np.array(bounds)
    return (np.array(point) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

# Calculate the parameter error
def calculateDistanceError(predict_points, real_extreme_point, input_data,origin_bounds, mode=0):
    error_array = []


    normalized_real_extreme_point = normalize_point(real_extreme_point, origin_bounds)
    normalized_predict_points = [normalize_point(point, origin_bounds) for point in predict_points]
    # distance error
    if mode == 0:
        for point in normalized_predict_points:
            
            error = np.linalg.norm(
                np.array(point) - np.array(normalized_real_extreme_point))
            error_array.append(error)

    else:
        # result error 
        data = input_data.Z 
        real_extreme_value = getRealValueFromData(real_extreme_point, data)

        for point in predict_points:

            pred_value = getRealValueFromData(point, data)
            print("real_value:",pred_value)

            max_value = np.max(data)
            min_value = np.min(data)
            print(max_value)
            print(min_value)
            print("absolute",(max_value-min_value))
            error = (np.abs(pred_value - real_extreme_value))/(max_value-min_value)
            error_array.append(error)
  
    return error_array

# Get real elevation from map dataset
def getRealValueFromData(point, data):
    real_value = data
    for index in point:
        int_index = int(round(index))
        real_value = real_value[int_index]
    return real_value


# Convergence determination
def lastNPointsConverged(epsilon, predict_points, origin_bounds, n=3):
    if len(predict_points) < n + 1:
        # Not enough points to compare
        return False
  
    min_bounds = [bound[0] for bound in origin_bounds]
    max_bounds = [bound[1] for bound in origin_bounds]


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([min_bounds, max_bounds]))

   
    scaled_predict_points = scaler.transform(predict_points)
   
    previous_error = float('inf')
    for i in range(1, n + 1):
        distance = np.linalg.norm(
            scaled_predict_points[-i] - scaled_predict_points[-i-1])
        # if distance >= epsilon or distance >= previous_error:
        if distance >= epsilon:
            return False
        previous_error = distance

    return True



# Calculate metrics
def calculate_metrics(model, poly, scaler, val_values, val_results):
 
    val_values_scaled = scaler.transform(val_values)

    val_poly_features = poly.transform(val_values_scaled)

   
    predictions = model.predict(val_poly_features)

  
    rmse = np.sqrt(mean_squared_error(val_results, predictions))
    
   
    r2 = r2_score(val_results, predictions)

  
    epsilon = 1e-8
    mape = np.mean(np.abs((val_results - predictions) / (val_results + epsilon))) * 100
    
   
    n = len(val_results)
    k = val_poly_features.shape[1] - 1  
    adjusted_r2 = r2 - ((1 - r2) * k) / (n - (k + 1))
    return rmse, r2, mape, adjusted_r2


# Extract samples on boundries in map datasets
def extract_outer_layer(array, mode='None'):
    
    shape = array.shape
    num_dims = len(shape)

    if mode == 'ignore_last':
        dimensions = num_dims - 1
    else:
        dimensions = num_dims

    outer_indices = set()

 
    for axis in range(dimensions):
        indices = np.indices(shape[:dimensions]).reshape(dimensions, -1).T

      
        indices_first = indices.copy()
        indices_first[:, axis] = 0
        outer_indices.update(map(tuple, indices_first))

       
        indices_last = indices.copy()
        indices_last[:, axis] = shape[axis] - 1
        outer_indices.update(map(tuple, indices_last))

    
    outer_indices = np.array(list(outer_indices))

    if mode == 'ignore_last':
        if num_dims > 1:
            outer_elements = array[tuple(outer_indices.T)] if num_dims > 2 else array[outer_indices[:, 0], outer_indices[:, 1], :]
        else:
            outer_elements = array[tuple(outer_indices.T)]
    else:
        outer_elements = array[tuple(outer_indices.T)]

    return outer_elements

# Unzip data folder
def unzip():
    if not os.path.exists('data/taranaki_detail5120.txt'):
        with zipfile.ZipFile('data.zip', 'r') as zip_ref:
            zip_ref.extractall()  
        print("unzip")
    else:
        print("files exist")
        
        


def generate_pairwise_inputs(variables_name, origin_bounds, precisions, constants_map, factors, default_value,isCluster=False):
    pairwise_inputs = []
    
    pairs = list(itertools.combinations(enumerate(variables_name), 2))

    for (idx1, var1), (idx2, var2) in pairs:
        pair_bounds = [origin_bounds[idx1], origin_bounds[idx2]]
        pair_variables_name = [var1, var2]
        pair_precisions = [precisions[idx1], precisions[idx2]]
        
        pair_factors = {k: factors[k] for k in pair_variables_name if k in factors}
        
        pair_constants_map = constants_map.copy()
        for var in variables_name:
            if var not in pair_variables_name:
                pair_constants_map[var] = constants_map.get(var, default_value.get(var, 0.0))
        
        pair_info = DimensionInfo(pair_variables_name, pair_constants_map)

        factor_array = None
        if not isCluster:
            pair_info.factors = factors
            if pair_factors:  
                factor_array = getMultiplyFactor(pair_variables_name, pair_factors)
                pair_bounds = getBoundsMultiplied(pair_bounds, factor_array, pair_precisions)
                
        else:
            pair_factors = {}  

        input_data = InputData(input_array=None, Z=None, precisions=pair_precisions,
                               fileName="bf_MD.csv", dimesion_info=pair_info,
                               origin_bounds=np.array(pair_bounds),
                               real_extreme_point=np.array([pair_bounds[0][1], pair_bounds[1][1]]),
                               factor_array=factor_array)
        pairwise_inputs.append(input_data)
    
    return pairwise_inputs

def apply_factors_to_default_values(default_value, factors):
    scaled_default_value = {}
    
    for key, value in default_value.items():
        if key in factors:
            scaled_value = value * factors[key][0] 
        else:
            scaled_value = value  
        scaled_default_value[key] = scaled_value
    
    return scaled_default_value

def generate_pairwise_inputs_3D(isCluster = False):
   
    variables_name = ['CokeOreRatio', 'HotBlastRate', 'f1']
    origin_bounds = [(1, 9), (10, 150), (2e-8, 1.9e-7)]
    precisions = [10, 10, 15]
    factors = {"f1": [1e7, 1e-7]}

    constants_map = {
        "f2": 0.0000001, "f3": 0.0000001, "f4": 0.0000001,
        "f5": 0.0000001, "f6": 0.0000001, "f7": 0.0000001
    }
    default_value = {"CokeOreRatio": 1, "HotBlastRate": 10, "f1": 0.0000001}
    scaled_default_value = apply_factors_to_default_values(default_value,factors)
    pairwise_inputs = generate_pairwise_inputs(variables_name, origin_bounds, precisions, constants_map, factors,scaled_default_value,isCluster)

    
    return pairwise_inputs

def setOptimisedConstant(lastMapInfo, curentMapInfo, predict_points):
 
    last_variables = lastMapInfo.dimesion_info.variables_name

    
    current_constants_map = curentMapInfo.dimesion_info.constants_map

    for var, value in zip(last_variables, predict_points):
        if var in current_constants_map:
            current_constants_map[var] = value


    curentMapInfo.dimesion_info.constants_map = current_constants_map
    print("curentMapInfo")
    print(curentMapInfo)

def getCombinedResult(curentMapInfo, predict_point):
    variable_names = curentMapInfo.dimesion_info.variables_name
    
    combined_result = {}
    
    for var, value in zip(variable_names, predict_point):
        combined_result[var] = value
    
    constants_map = curentMapInfo.dimesion_info.constants_map
    for var, value in constants_map.items():
        combined_result[var] = value
    print(combined_result)
    return combined_result


result = generate_pairwise_inputs_3D()
print(result[0])
# setOptimisedConstant(result[0],result[1],(9,150))
# print(result[1])
# combined_result = getCombinedResult(result[0],(9,150))
# print(len(combined_result))