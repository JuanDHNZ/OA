# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 08:11:23 2021

@author: dcard
"""
import numpy as np
from scipy.signal import butter, filtfilt
from mne.decoding import CSP

class FBCSP():     
    def __init__(self,fs,f_low,f_high,bandwidth,step,n_components=4):
        self.n_components = n_components
        self.fs = fs
        self.bands = [np.array([low,low+bandwidth]) for low in np.arange(f_low,f_high-bandwidth+1,step)]
        return    
    
    def set_params(self,fs,f_low,f_high,bandwidth,step,n_components=4):
        self.n_components = n_components
        self.fs = fs
        self.bands = [np.array([low,low+bandwidth]) for low in np.arange(f_low,f_high-bandwidth+1,step)]
        return
    
    def fit(self,X,y):
        self.csp = []
        nyq = 0.5 * self.fs
        
        self.labels = np.unique(y)
        
        for band in self.bands:
            b,a= butter(5, band/nyq, btype='band')
            Xband = filtfilt(b,a, X, axis=2) #trials x ch x time
            csp_band = []
            for c in self.labels:
                csp_class = CSP(self.n_components).fit(Xband,y==c)
                csp_band.append(csp_class)
            self.csp.append(csp_band) #[banda][clase]
        return self
    
    def transform(self,X):
        feats = []
        nyq = 0.5 * self.fs
        bands = self.bands
        for i in range(len(self.bands)):
            band = bands[i]
            b,a= butter(5, band/nyq, btype='band')
            Xband = filtfilt(b,a, X, axis=2)
            for csp_class in self.csp[i]:
                feats.append(csp_class.transform(Xband)) #i,1; i,2; i,3; i,4; i+1,1; i+1;2
        return np.concatenate(feats,axis=1)
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
    