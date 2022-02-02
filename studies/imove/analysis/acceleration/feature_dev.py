# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq


# Import own modules
from acceleration import resample, align_timestamp
from paths import path_data

# LOAD iMove_Borg_JB.csv 
filepath = Path(path_data,'Borg/iMove_Borg_JB.csv')
iMove_Borg_JB = pd.read_csv(filepath)
borg = iMove_Borg_JB[["patient_ID", "sex", "age", "weight", "BMI"]]
borg.loc[:,'patient_ID'] = borg['patient_ID'].apply(str) # int -> str
borg.loc[:,'patient_ID'] = borg['patient_ID'].str.rjust(3, "0") # add trailing zeros
# borg.info()

def get_patient_mass(patient_ID):
    """for specific patient_ID, return mass, ie weight."""
    mask = borg.patient_ID==patient_ID # list(g.groups.keys())[0]  # eg patient_ID = "003"
    print(mask)
    mass = borg.loc[mask, "weight"]
    return mass

# a ----------------------------------------------------------------------------
def feature_development(df, ex='12'):
    """Index of df: RangeIndex
    """
    # subset input df (without margins) for specific ex 
    mask = df.DeMortonLabel.eq(ex)
    df = df[mask]
        
    # 1)
    def score_std(df):
        """input df is groupby object g. Compute Std of A per group."""
        df = align_timestamp(df=df, grouping=['Patient', 'DeMortonDay', 'Side'])

        def transform(df): 
            """to be applied on each subgroup. Calculate Standard deviation of A."""
            A_std = df["A"].std() # Series
            return A_std

        # groupby Patient, Side, DeMortonDay                                                                                                                   
        g = df.groupby(["Patient", "DeMortonDay", "Side"])
        
        scores_std = g.apply(transform)
        scores_std.name = "Standard deviation"
        
        return scores_std

    
    # 2)
    def score_kinetic_energy(df):
        """input df is groupby object g. 
        Compute kinetic energy given acceleration and mass"""
        
        df = align_timestamp(df=df, grouping=['Patient', 'DeMortonDay', 'Side'])

        # Make col 'time' the new Index (Int64Index is no more index)
        df = df.set_index('time') # needed for resample() later
        
        def transform(df): # warum müssen wir resamplen?
            """to be applied on each subgroup. resample A. Calculate E_kin_total.
            f(A) >> v
            f(v, get_patient_mass(patient_ID)) >> Ek """

            dt = 1 # 0.5 # Time step in seconds
            v0 = 0 #  assumption for velocity at time 0 
            m = 60
                              
            ## Method 1: Per patient, day, exercise and side
            A = df["A"] 
            a_filt = A.resample(rule='1min').mean()
            v_filt = a_filt.cumsum()*dt + v0 # cumsum() beginnt bei 0 implizit
            E_kin = m*v_filt**2
            E_kin_total = E_kin.sum() # warum aufsummieren?
            
            return E_kin_total
        
        # groupby Patient, Side, DeMortonDay                                                                                                                   
        g = df.groupby(["Patient", "DeMortonDay", "Side"])

        scores_kin = g.apply(transform)
        scores_kin.name = "Kinetic energy"
        return scores_kin


    # 3)
    def score_spectrum(df):
        """Compute FFT of A. 
        Output: Mean frequency, ie (xf * yf_norm).mean() """
        df = align_timestamp(df=df, grouping=['Patient', 'DeMortonDay', 'Side'])

        def transform(df): 
            """to be applied on each subgroup. Calculate FFT of A"""
            SAMPLE_RATE = 51.2 # Hz
        
            # Fast Fourier transform (FFT) of acceleration A
            u = df["A"]
            
            # Eliminate static component of g, with this simple method
            u -= u.mean()  # subtract mean to each A-value (u is Series)
            
            # xf: Array of float64 
            # yf: Array of complex64
            yf = rfft(x = u.values) # compute 1-D n-point discrete Fourier Transform (DFT) of a real-valued array. (u.values is numpy.ndarray).
            xf = rfftfreq(n=len(u), d=1/SAMPLE_RATE) # calculate frequencies in center of each bin. 
            
            # Euclidean Norm of yf (complex values)
            yf_norm = np.abs(yf) # cf: https://stackoverflow.com/questions/62983674/the-absolute-value-of-a-complex-number-with-numpy

            return xf[np.argmax(yf_norm)] # frequenz von grösstem Peak. # Methode JB: (xf * yf_norm).mean()
            # save plot as: https://github.com/hirsch-lab/mhealth/blob/feature/tm2_julien/studies/imove/analysis/analysis_demorton.py
            # save_figure() verwenden. (innerhalb v score_spectrum())

        # groupby Patient, Side, DeMortonDay                                                                                                                   
        g = df.groupby(["Patient", "DeMortonDay", "Side"])
    
        scores_kin = g.apply(transform)
        scores_kin.name = "Kinetic energy"
        return scores_kin


    ## Execute all inner functions
    scores_std   = score_std(df)
    scores_kin   = score_kinetic_energy(df)
    scores_spect = score_spectrum(df)
    
    # scores_all: DataFrame with MultiIndex: Patient, DeMortonDay, Side; and 2 cols: "Standard deviation" and "Characteristic MEAN frequency"
    scores_all = pd.concat([ # concat all Series
                            scores_std, 
                            scores_kin,
                            scores_spect
                            ], axis=1)
    return scores_all
    
    


a = feature_development(df=acc, ex='12')
