# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:43:10 2021

@author: Juan David
"""
import numpy as np
# from mac import MAC
from sklearn.decomposition import FastICA
from KAF import QKLMS
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kurtosis
import KAF

class ar_KLMS():
    """
    Artifact remotion from EEG signals based on EOG reference signals
    """
    def __init__(self, eog_ref = [0,1]):
        self.eog_ref = eog_ref
        return
    
    def get_params(self, deep=True):
        return {"eog_ref": self.eog_ref}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        return
    
    def transform(self, X):
        EOG_set = X[:,self.eog_ref,:]   # trials x ch x time
        EEG_set = np.delete(X,self.eog_ref,axis=1) # trials x ch x time
        
        #1. Noise estimation x trial
        noiseXtrial = []
        cleanEEG = []
        for eeg,eog in zip(EEG_set,EOG_set):
            # eeg and eog of shape ch x time
            c,t = eeg.shape
            cn,tn = eog.shape
            
            ica = FastICA(n_components=c, max_iter=500)
            S = ica.fit_transform(eeg.T).T  # ch x time
            
            corr = np.corrcoef(S.T, eog.T, rowvar=False)
            corr = corr[:c,c:]        
            mac = np.abs(corr).mean(axis=1)
            ic_noise_index = mac.argmax()
            
            noise = S[ic_noise_index]  # Top noise-like IC
            noiseXtrial.append(noise)
            
            
            #2. Artifact remotion
            clean_trial = self.__noise_removal(eeg,noise) # eeg (22,500) & noise (500,)
            cleanEEG.append(clean_trial)
        return np.array(cleanEEG)
    
    def fit_transform(self, X,y):
        return self.transform(X)
                    
    def __noise_removal(self, trial, ch_noise):
        """
        trial -> EEG trial of shape (ch,time)
        ch_noise -> Noise-like IC of shape (time,)
        """
        return np.array([self.__ch_noise_removal(t,ch_noise) for t in trial])
        
    def __ch_noise_removal(self, ch, ch_noise):
        """
        ch -> EEG trial channel of shape (time,)
        ch_noise -> Noise-like IC of shape (time,)
        """
        embedding = 5
        noise_em = np.array([ch_noise[i-embedding:i] for i in range(embedding,len(ch_noise))])
        X_em = np.array([ch[i] for i in range(embedding,len(ch))]).reshape(-1,1)
        
        af = QKLMS(epsilon = 0)
        noise_pred = af.evaluate(noise_em,X_em)
        return X_em.ravel() - noise_pred.ravel()
    
    
class ar_KLMS_qt():
    """
    Prueba rapida - Pendiente borrar si no se necesita
    """
    def __init__(self, eog_ref = [0,1]):
        self.eog_ref = eog_ref
        return
    
    def get_params(self, deep=True):
        return {"eog_ref": self.eog_ref}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        return
    
    def transform(self, X):
        EOG_set = X[:,self.eog_ref,:]   # trials x ch x time
        EEG_set = np.delete(X,self.eog_ref,axis=1) # trials x ch x time
        
        #1. Noise estimation x trial
        noiseXtrial = []
        cleanEEG = []
        for eeg,eog in zip(EEG_set,EOG_set):
            
            noise = eog[1]

            #2. Artifact remotion
            clean_trial = self.__noise_removal(eeg,noise) # eeg (22,500) & noise (500,)
            cleanEEG.append(clean_trial)
        return np.array(cleanEEG)
    
    def fit_transform(self, X,y):
        return self.transform(X)
                    
    def __noise_removal(self, trial, ch_noise):
        """
        trial -> EEG trial of shape (ch,time)
        ch_noise -> Noise-like IC of shape (time,)
        """
        return np.array([self.__ch_noise_removal(t,ch_noise) for t in trial])
        
    def __ch_noise_removal(self, ch, ch_noise):
        """
        ch -> EEG trial channel of shape (time,)
        ch_noise -> Noise-like IC of shape (time,)
        """
        embedding = 5
        noise_em = np.array([ch_noise[i-embedding:i] for i in range(embedding,len(ch_noise))])
        X_em = np.array([ch[i] for i in range(embedding,len(ch))]).reshape(-1,1)
        
        af = QKLMS(epsilon = 0)
        noise_pred = af.evaluate(noise_em,X_em)
        return X_em.ravel() - noise_pred.ravel()
    
class AR_ICA():
    """
    Artifact remotion methodology using ICA
    """
    def __init__(self, threshold = 1, ch_ref = [22,23,24]):
        self.threshold = threshold
        self.ch_ref = ch_ref
        return
    
    def get_params(self, deep=True):
        return {"threshold": self.threshold, "ch_ref":self.ch_ref}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        return
    
    def transform(self, X):
        Xeeg = np.delete(X,self.ch_ref,axis=1)
        Xclean = []
        k4trial = []
        for trial in tqdm(Xeeg):
            chs, t = trial.shape
            
            ica = FastICA(n_components=chs)
            S = ica.fit_transform(trial.T).T  # ch x time
            K = kurtosis(S, axis = 1).reshape(-1) # ch
            k4trial.append(K)
            index = np.where(K < self.threshold)[0]
            
            A = ica.mixing_
            At = A[:,index]
            St = S[index]
            clean_trial = np.dot(At, St)

            Xclean.append(clean_trial)
        print("error = ", np.linalg.norm(Xeeg - Xclean)/np.linalg.norm(Xeeg))
        self.kurtosis = np.array(k4trial)
        return np.array(Xclean)
    
    def fit_transform(self, X,y):
        return self.transform(X)    

class AR_ICA_KLMS():
    """
    Artifact remotion methodology using ICA
    """
    def __init__(self, threshold = 1, ch_ref = [22,23,24], signalEmbedding = 5):
        self.threshold = threshold
        self.ch_ref = ch_ref
        self.signalEmbedding = signalEmbedding
        return
    
    def get_params(self, deep=True):
        return {"threshold": self.threshold, "ch_ref":self.ch_ref}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        return
    
    def transform(self, X):
        Xeeg = np.delete(X,self.ch_ref,axis=1)
        Xclean = []
        k4trial = []
        for trial in tqdm(Xeeg):
            chs, t = trial.shape
            
            ica = FastICA(n_components=chs)
            S = ica.fit_transform(trial.T).T  # ch x time
            K = kurtosis(S, axis = 1).reshape(-1) # ch
            k4trial.append(K)
            index = np.where(K > self.threshold)[0]
            if len(index) > 0:
                A = ica.mixing_
                At = A[:,index]
                St = S[index]
                noise = np.dot(At, St)
                noise += ica.mean_.reshape(-1,1)

                clean_trial = [self.__ch_noise_removal(n,tr) for tr,n in zip(trial,noise)]
                Xclean.append(clean_trial)
            else:
                Xclean.append(trial[:,self.signalEmbedding:])

        print("error = ", np.linalg.norm(Xeeg[:,:,self.signalEmbedding:] - Xclean)/np.linalg.norm(Xeeg))
        self.kurtosis = np.array(k4trial)
        return np.array(Xclean)
    
    def fit_transform(self, X,y):
        return self.transform(X)  

    def __ch_noise_removal(self, noise_true, X_true):
        import numpy as np

        noise_em = np.array([noise_true[i-self.signalEmbedding:i] for i in range(self.signalEmbedding,len(noise_true))])
        X_em = np.array([X_true[i] for i in range(self.signalEmbedding,len(noise_true))]).reshape(-1,1)
        
        from KAF import QKLMS
        f = QKLMS(epsilon = 0)
        noise_pred = f.evaluate(noise_em,X_em)
        return X_em.ravel() - noise_pred.ravel()
