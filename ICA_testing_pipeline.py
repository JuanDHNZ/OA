# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:42:34 2021

@author: Juan David

ICA pipeline: 
    Test with RAW data

    PIPELINE      ---->     ICA/kurtosis -> FBCSP -> SelectKBest -> SVC
    
    threshold from AR_ICA, k from selectKBest & C from SVC were tuned

"""

import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from fbcsp import FBCSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score,make_scorer
from artifact_removal import AR_ICA, AR_ICA_KLMS


parser = argparse.ArgumentParser(description='FBCSP -> SelectKBest -> SVC')
parser.add_argument('--train',required=True, help='Input train filename with path')
parser.add_argument('--test',required=True, help='Input test filename with path')
parser.add_argument('--params',required=True, help='Parameters search results file')

args = parser.parse_args()

train_file = args.train
test_file = args.test
params_file = args.params

# 1. load BCI data train
data = sio.loadmat(train_file)
X_train = data['X']
y_train = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',train_file,'with sampling frequency of',fs,'Hz.')
print("T R A I N I N G . . .")
print("Train set size:\nXtrain: {} \t ytrain: {}".format(X_train.shape, y_train.shape))

# 2. Parameter selection
parameters = pd.read_csv(params_file)
bp = parameters[parameters.mean_test_score == parameters.mean_test_score.max()].iloc[0]

# 3. Pipeline definition and trainnig
steps = [ ('preproc', AR_ICA_KLMS(threshold=bp.param_preproc__threshold)), #AR_ICA(threshold=bp.param_preproc__threshold)
          ('extract', FBCSP(fs,4,40,4,4,n_components=4)),
          ('select', SelectKBest(k=bp.param_select__k, score_func=mutual_info_classif)),          
          ('classify',SVC(C=bp.param_classify__C, kernel='linear'))
        ]

pipeline = Pipeline(steps = steps)
pipeline.fit(X_train,y_train) 

# 4. Prediction
data = sio.loadmat(test_file)
X_test = data['X']
y_test = data['labels'].reshape(-1,)
fs = int(data['fs'].reshape(-1,))
print('Loading',test_file,'with sampling frequency of',fs,'Hz.')
print("T E S T I N G . . .")
print("Test set size:\nXtrain: {} \t ytrain: {}".format(X_test.shape, y_test.shape))

y_pred = pipeline.predict(X_test)
kappa_corr = lambda target,output : (cohen_kappa_score(target,output)+1)/2
score = kappa_corr(y_test,y_pred)
print('\nSCORE = ', score)
print('\n\n')






