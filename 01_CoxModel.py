#%%

import pandas as pd
import numpy as np

from sklearn.feature_selection import RFECV
from sksurv.linear_model import CoxPHSurvivalAnalysis
from _crossval import StratifiedKFold_surv
from eli5.sklearn import PermutationImportance

#%% Load data for outer cross-validation folds (CV2)

data_1=pd.read_csv(r"data\data_1.csv",index_col=0)
data_2=pd.read_csv(r"data\data_2.csv",index_col=0)
data_3=pd.read_csv(r"data\data_3.csv",index_col=0)
data_4=pd.read_csv(r"data\data_4.csv",index_col=0)

data_1.e=data_1.e.astype(bool)
data_2.e=data_2.e.astype(bool)
data_3.e=data_3.e.astype(bool)
data_4.e=data_4.e.astype(bool)

#%% Run cross-validation framework

nPerm2=5
kCV2=4

nPerm1=1
kCV1=4


feature_importances_Perm2=[]
predictions_Perm2=[]

for Perm2 in range(1,nPerm2+1):

    print("Perm2: ",Perm2)

    feature_importances_CV2=[]
    predictions_CV2=[]

    for CV2 in range(1,kCV2+1):

        print("CV2: ",CV2)

        if CV2==1:
            train_df=pd.concat([data_2,data_3,data_4],ignore_index=True)
            test_df=data_1
        elif CV2==2:
            train_df=pd.concat([data_1,data_3,data_4],ignore_index=True)
            test_df=data_2
        elif CV2==3:
            train_df=pd.concat([data_1,data_2,data_4],ignore_index=True)
            test_df=data_3
        elif CV2==4:
            train_df=pd.concat([data_1,data_2,data_3],ignore_index=True)
            test_df=data_4


        train_X=train_df.iloc[:,:-2]

        train_y=train_df.iloc[:,-2:]
        tmp = train_y.dtypes
        train_y = np.array([tuple(x) for x in train_y.values], dtype=list(zip(tmp.index, tmp)))

        del tmp

        test_X=test_df.iloc[:,:-2]

        test_y=test_df.iloc[:,-2:]
        tmp = test_y.dtypes
        test_y = np.array([tuple(x) for x in test_y.values], dtype=list(zip(tmp.index, tmp)))

        del tmp


        estimator = CoxPHSurvivalAnalysis()
        selector = RFECV(estimator,cv=StratifiedKFold_surv(n_splits=kCV1, shuffle=True), verbose=1, n_jobs=-1)

        selector.fit(train_X,train_y)
        predictions=selector.predict(test_X)

        test_X_sel=test_X.iloc[:,selector.support_]

        permI = PermutationImportance(selector.estimator_, n_iter=5, random_state=42)
        permI.fit(test_X_sel, test_y)

        feature_importances=np.zeros((test_X.shape[1],),dtype=float)
        feature_importances[selector.support_]=permI.feature_importances_

        predictions_CV2.append(predictions)
        feature_importances_CV2.append(feature_importances)


    predictions_CV2=np.asarray(predictions_CV2)
    predictions_CV2=np.concatenate((predictions_CV2[0],predictions_CV2[1],predictions_CV2[2],predictions_CV2[3]))

    feature_importances_CV2=np.mean(feature_importances_CV2,axis=0)

    predictions_Perm2.append(predictions_CV2)
    feature_importances_Perm2.append(feature_importances_CV2)


predictions_Perm2=np.mean(predictions_Perm2,axis=0)
feature_importances_Perm2=np.mean(feature_importances_Perm2,axis=0)

#%% Export predictions

e=np.concatenate((data_1.e,data_2.e,data_3.e,data_4.e))
t=np.concatenate((data_1.t,data_2.t,data_3.t,data_4.t))

e=pd.DataFrame(e.astype(int),columns=['e'])
t=pd.DataFrame(t,columns=['t'])

predictions_CoxModel=pd.DataFrame(predictions_Perm2,columns=['predictions_CoxModel'])

predictions_export=pd.concat([predictions_CoxModel,e,t],axis=1)
predictions_export.to_csv("predictions_CoxModel.csv")

#%% Export permutation importance 

feature_importances_export=pd.Series(feature_importances_Perm2, index=train_X.columns).sort_values(ascending=False)
feature_importances_export.to_csv("feature_importances_CoxModel.csv",header=False)

# %%
