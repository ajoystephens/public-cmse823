# Script for analyzing error between our appraoch and sklearn PCA

import numpy as np
import pandas as pd
import seaborn as sns
import random
import sys

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

sys.path.append('lib/')
from NumericPCA import NumericPCA

N = 50 # number of rows in test matrix
M = 10 # number of columns in test matrix

TRIALS = 100 # number of trials
N_COMPONENTS = [2,4,6,8]

RESULTS_FILEPATH_FULL = 'results/full.csv'
RESULTS_FILEPATH_SUMMARY = 'results/summary.csv'
RESULTS_FILEPATH_PLOT = 'results/plot.png'

# function to calc error
def getReconstructionError(ogData,newData):
    rmse = np.sqrt(mean_squared_error(ogData, newData))
    # nrmse = rmse/np.sqrt(np.mean(ogData**2))
    # diff = np.abs(ogData - newData)
    return(rmse)




results = pd.DataFrame(columns=['trial','n_components','method','rmse'])

for t in range(TRIALS):
    X = np.random.randint(1,10,size=(N,M))
    
    for p in N_COMPONENTS:
        ignore_run = False
        nPCA = NumericPCA(n_components=p)
        fit = nPCA.fit(X)
        X_trans = nPCA.transform(X)
        X_new = nPCA.inverse_transform(X_trans)
        if np.any(np.isnan(X_new)): ignore_run = True
        # print(X_new)
        if not ignore_run: nPCA_error = getReconstructionError(X,X_new)
        
        
        pca = PCA(n_components=p)
        fit = pca.fit(X)
        X_trans = pca.transform(X)
        X_new = pca.inverse_transform(X_trans)
        if np.any(np.isnan(X_new)): ignore_run = True
        # print(X_new)
        if not ignore_run: pca_error = getReconstructionError(X,X_new)
        
        # print(ignore_run)
        if not ignore_run:
            r = [{'trial': t,'n_components': p,'method': 'project','rmse': nPCA_error,},
                {'trial': t,'n_components': p,'method': 'sklearn','rmse': pca_error,}]
            results = results.append(r, True)
            # print(r)
            
            # print(results.head())
            
results.to_csv(RESULTS_FILEPATH_FULL,index=False)

summary = results.groupby(['n_components','method']).agg({'rmse': ['mean', 'std'],}).reset_index()
summary.to_csv(RESULTS_FILEPATH_SUMMARY,index=False)

print(summary)


plot = sns.boxplot(y='rmse', x='n_components', 
                 data=results, 
                 hue='method')
fig = plot.get_figure()
fig.savefig(RESULTS_FILEPATH_PLOT)