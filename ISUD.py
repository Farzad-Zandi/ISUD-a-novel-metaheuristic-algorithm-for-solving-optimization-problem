# Farzad Zandi, 2024.
# ISUD (Individuals with Substance Use Disorder): a novel metaheuristic algorithm for solving optimization problem.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Copy function.py file in the same path with ISUD.py for access to benchmark function.
from Functions import *

print("=====================")
print("Farzad Zandi, 2024...")
print("ISUD (Individuals with Substance Use Disorder): a novel metaheuristic algorithm for solving optimization problem...")

# Set the parameters.
lb = -5  # Lower bound.
ub = 5 # Upper bound.
n = 10 # Dimension.
nISUD = 100 # Number of ISUD.
maxIter = 500 # Maximum iteration.
X = np.random.uniform(low= lb, high= ub, size=(nISUD, n)) # Initializing Positions.
fit = np.full(nISUD, np.inf) # Initializing Fitness.
M = X # Memory.
r = 0.7 # Ratio of Old to New ISUD.
idx = np.random.permutation(nISUD)
idxOld = idx[0:int(r*nISUD)] # Determining Old ISUD.
idxNew = idx[int(r*nISUD):nISUD] # Determining New ISUD.
globalFit = np.inf # Initialize global fitness.
sT = np.full(nISUD, np.inf) # Initializing search time.
alpha = 0.9 # Initializing time reduce Coefficient.
conv = maxIter / 10 # Initializing conversion rate, New to Old converting.
yPlot = np.full(maxIter, np.inf) # Plotting parametter.

# Calculating fitness.
# Replace F1 with other function.
for i in range(nISUD):
    fit[i] = F1(X[i,:])

globalFit = bestFit = min(fit)
idx = np.argmin(fit)
xHat = X[idx,:]

# Initializing search time.
for i in idxNew:
    sT[i] = np.random.uniform(low= -2, high=2)
for i in idxOld:
    sT[i] = np.random.uniform(low= -1, high=1)  

for t in range(maxIter):
    # Converting a New random ISUD to an Old.
    if (t+1) % conv == 0:
        j = np.random.permutation(idxNew)[0]
        idxOld = np.append(idxOld, j) 
        j = np.where(idxNew==j)
        idxNew = np.delete(idxNew, j)
    # Updating the positions.    
    for i in idxNew:
        j = np.random.permutation(idxOld)[0]
        rand = np.random.random()
        xNew = X[i,:] + rand * sT[i] * (M[j,:] - X[i,:])    
        X[i,:]  = np.clip(xNew, lb, ub)
    for i in idxOld:
        if np.mean(fit)<fit[i] or (sT[i]<0.1 and sT[i]>-0.1):
            rand = np.random.random()
            xNew = X[i,:] + rand * sT[i] * X[i,:]
            X[i,:]  = np.clip(xNew, lb, ub)
        else:
            j = np.random.permutation(idxNew)[0]
            rand = np.random.random()
            xNew = X[i,:] + rand * sT[i] * (M[j,:] - X[i,:])   
            X[i,:]  = np.clip(xNew, lb, ub)
    # Calculating fitness of new positions and Update the positions and memories.
    for i in range(nISUD):    
        y = F1(X[i,:])
        if fit[i] > y:
            M[i,:] = X[i,:]
        fit[i] = y    
    # Setting the global fitness. 
    bestFit = min(fit)
    idx = np.argmin(fit)
    if globalFit>bestFit:
        xHat = X[idx,:]
        globalFit = bestFit
    print(f"Best fitness in iteration: {t+1} is: {globalFit}")  
    yPlot[t] = globalFit
    # Decresing serach time.
    sT = alpha * sT   
print(xHat)  
print(f"{globalFit: 1e}")  

plt.plot(range(maxIter), yPlot, label= 'Best fitness= %1e' %(globalFit))
plt.title("ISUDA Convergence")
plt.ylabel("Fitness value")
plt.xlabel("Iteration")
plt.legend()
plt.grid(True)
plt.show()
