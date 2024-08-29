import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
import time
import random
from utilities import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import differential_evolution
from pyswarm import pso
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# train polynomial regression model
def predict(sampled_values, sampled_results, the_degree=2, bounds=None, log_transform=True):
    scaler = generateScaler(bounds)

    X = sampled_values

    y = sampled_results

    X_scaled = scaler.transform(X) 

    poly = PolynomialFeatures(degree=the_degree)
    X_train_poly = poly.fit_transform(X_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y)

    return model, poly, scaler

# Object fucntion get from trained polynomial model
def objective_function(inputs, model, poly, scaler, findMax=True):
  
    inputs_scaled = scaler.transform([inputs])
  
    inputs_poly = poly.transform(inputs_scaled)
  
    if findMax:
        return -model.predict(inputs_poly)[0]
    else:
        return model.predict(inputs_poly)[0]

# uses PSO (‘pyswarm’) or COBYLA (‘scipy’) to get the optimal input parameter combination 
# within the search space, based on the objective function
# provided by the trained model.
def findOptimum(model, poly, scaler, bounds, findMax=True, random_guess=1, method='COBYLA'):
    if method == 'COBYLA':
        print('COBYLA')
        return findOptimumUsingCOBYLA(model, poly, scaler, bounds, findMax, random_guess, method)
    else:
        return findOptimumUsingPSO(model, poly, scaler, bounds, findMax, random_guess, method)
    
def findOptimumUsingCOBYLA(model, poly, scaler, bounds, findMax=True, random_guess=1, method='Powell'):
    num_dimensions = len(bounds)
    min_bounds = [bound[0] for bound in bounds]
    max_bounds = [bound[1] for bound in bounds]

    # Init initial guess
    guesses = np.random.rand(random_guess, num_dimensions)
    for i in range(num_dimensions):
        guesses[:, i] = guesses[:, i] * \
            (max_bounds[i] - min_bounds[i]) + min_bounds[i]

    results = []
    points = []

    def opt_function(inputs):
        return objective_function(inputs, model, poly, scaler, findMax)
    # Try several times to avoid trapped in local optimum
    for guess in guesses:
        try:
            result = minimize(opt_function, guess, bounds=[
                              (min_bounds[i], max_bounds[i]) for i in range(num_dimensions)], method=method)
            points.append(result.x)
            results.append(-result.fun if findMax else result.fun)
        except Exception as e:
            print(f"Optimization failed for initial guess {guess}: {e}")

    return points, results

def findOptimumUsingPSO(model, poly, scaler, bounds, findMax=True, random_guess=1, method=None):
    num_dimensions = len(bounds)
    min_bounds = [bound[0] for bound in bounds]
    max_bounds = [bound[1] for bound in bounds]

   
    def opt_function(inputs):
        return objective_function(inputs, model, poly, scaler, findMax)
    
    lb = np.array(min_bounds)
    ub = np.array(max_bounds)

    points = []
    results = []

   
    for _ in range(10):
        try:
            xopt, fopt = pso(opt_function, lb, ub, swarmsize=100, maxiter=200)
            points.append(xopt)
            results.append(-fopt if findMax else fopt)
        except Exception as e:
            print(f"PSO optimization failed: {e}")

    return points, results

# Use polynomial model to predict data ,just to visual the reconstructed contour plot
def reconstruct_map(model, poly, scaler, bounds, resolution=500):
 
    min_bounds = [bound[0] for bound in bounds]
    max_bounds = [bound[1] for bound in bounds]

  
    grid_points = np.meshgrid(*[np.linspace(min_bound, max_bound, resolution)
                              for min_bound, max_bound in zip(min_bounds, max_bounds)], indexing='ij')
    grid_points = np.column_stack([gp.ravel() for gp in grid_points])

  
    grid_points_scaled = scaler.transform(grid_points)

 
    grid_points_poly = poly.transform(grid_points_scaled)
    predicted_values = model.predict(grid_points_poly)

    
    coordinates = [np.linspace(min_bound, max_bound, resolution)
                   for min_bound, max_bound in zip(min_bounds, max_bounds)]
    reshaped_predicted_values = predicted_values.reshape(
        [resolution] * len(bounds))

    return coordinates, reshaped_predicted_values



# Not using
class CenterMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.pipeline = Pipeline([
            ('center', StandardScaler(with_mean=True, with_std=False)),
            ('minmax', MinMaxScaler(feature_range=self.feature_range))
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)

# Using MinMaxScaler
def generateScaler(bounds, scaler_type=1):

    min_bounds = [bound[0] for bound in bounds]
    max_bounds = [bound[1] for bound in bounds]

    if scaler_type == 1:
        print("MinMaxScaler")
       
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array([min_bounds, max_bounds]))
    elif scaler_type == 2:
        print("CenterMinMaxScaler")
       
        scaler = CenterMinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array([min_bounds, max_bounds]))
    else:
        print("StandardScaler")
       
        grid_points = np.meshgrid(*[np.linspace(min_bound, max_bound, 500)
                                  for min_bound, max_bound in bounds], indexing='ij')
        grid_points = np.array([gp.ravel() for gp in grid_points]).T

        
        scaler = StandardScaler()
        scaler.fit(grid_points)

    return scaler






