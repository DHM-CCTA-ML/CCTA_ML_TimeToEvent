#%%

import pandas as pd
import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#%%

cCI = importr('compareC')

def compareC(timeX, statusX, scoreY, scoreZ):
    stats=cCI.compareC(timeX, statusX.astype('int'), scoreY, scoreZ)
    return stats

def print_stats(stats):
    CI_Y=1-stats[0][0]
    CI_Z=1-stats[0][1]

    print("cIndex_Y =",CI_Y)
    print("cIndex_Z =",CI_Z)
    print("pval =",stats[7][0])
    print("ConfidenceInterval_lower_Y =",CI_Y-(1.96*np.sqrt(stats[3][0])))
    print("ConfidenceInterval_upper_Y =",CI_Y+(1.96*np.sqrt(stats[3][0])))
    print("ConfidenceInterval_lower_Z =",CI_Z-(1.96*np.sqrt(stats[4][0])))
    print("ConfidenceInterval_upper_Z =",CI_Z+(1.96*np.sqrt(stats[4][0])))

#%% Load predictions

predictions_CoxModel=pd.read_csv(r"predictions_CoxModel.csv", index_col=0)
predictions_MLModel=pd.read_csv(r"predictions_MLModel.csv", index_col=0)

#%% Compare cox model and machine learning model and print results

stats=compareC(predictions_CoxModel.t, predictions_CoxModel.e, predictions_CoxModel.predictions_CoxModel, predictions_MLModel.predictions_MLModel)
print_stats(stats)

# %%
