# FEATURE DEVELOPMENT

# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq


# Import own modules
from acceleration import resample, align_timestamp
from paths import path_data

# FEATURE DEVELOPMENT ----------------------------------------------------------------------------

# Und dann noch das Rezept für die für die Berechnung der Features.
# Angenommen, wir möchten verschiedene Features für verschiedene Übungen bestimmen:

# Selektiere die relevanten Daten (ohne Margin) für eine bestimmte üung
    # Gruppiere die Daten nach Patienten, nach Seite, nach Tag
    # Für die Data-Chunks, rufe Score-Funktionen auf, die du vorher definiert hast
    # Sammle den Output
    # Berechne Mittelwerte
    # Berechne Statistik über verschiedene Patienten
def feature_development(df, ex='12'):
    """Input df: acc. Subset specific ex. No resampling, keep 52 Hz.
    align_timestamp(). 
    Groupby: Patient, Day, Side.
    Define various score functions. Apply score functions per groups --> Series.
    Concatenate all Score series together -> scores_all
    Groupby Patient, calculate mean -> scores
    Output: scores (df)
    """
    
    # subset input df (without margins) for specific ex 
    mask = df.DeMortonLabel.eq(ex)
    df = df[mask]
    
    # Set timestamp as Datetimeindex
    df = df.set_index('timestamp')
    
    # apply resample() but with enable=False. Makes only reset_index(), which is needed.
    df = resample(df, enable=False) 
    
    # align all acc-curves with common starting time = 0
    df_aligned = align_timestamp(df=df)
    
    # groupby Patient, Side, DeMortonDay                                                                                                                   
    # DELETE THIS TEXXT AGAIN: Example: Compute a score for exercise 12
    g = df_aligned.groupby(["Patient", "DeMortonDay", "Side"])
    
    
    # DEFINE SCORE FUNCTIONS, that operate on the groupby object g:
    # Each score function generates 1 values per group in g
    
    # 1)
    def score_std(g):
        """input df is groupby object g. Compute Std of A per group."""
        # Standard deviation of A per group
        A_std = g["A"].std() # Series
        return A_std
    
    # 2) WEISS NICHT, WIE DIESES ZU IMPLEMENTIEREN IST
    def score_kinetic_energy(g, masses): # masses hier entfernen als Argument!
        """input df is groupby object g. 
        Compute kinetic energy given acceleration and mass"""

        # LOAD iMove_Borg_JB.csv 
        filepath = Path(path_data,'Borg/iMove_Borg_JB.csv')
        iMove_Borg_JB = pd.read_csv(filepath)
        borg = iMove_Borg_JB[["patient_ID", "sex", "age", "weight", "BMI"]]
        borg.loc[:,'patient_ID'] = borg['patient_ID'].apply(str) # int -> str
        borg.loc[:,'patient_ID'] = borg['patient_ID'].str.rjust(3, "0") # add trailing zeros
        # borg.info()
        
        ## borg LUT 
        # pat = "003"
        # mask = borg.patient_ID==pat
        # bmi = borg.loc[mask, "BMI"]
    
        def get_patient_BMI(patient_ID):
            """for specific patient_ID, return BMI."""
            mask = borg.patient_ID==patient_ID # eg patient_ID = "003"
            bmi = borg.loc[mask, "BMI"]
            
        def get_patient_mass(patient_ID):
            """for specific patient_ID, return mass, ie weight."""
            mask = borg.patient_ID==patient_ID # eg patient_ID = "003"
            mass = borg.loc[mask, "weight"]
            
    
        # get subgroup's Patient_ID
        # patient_ID = g.... # weiss nicht wie. Oder müssen wir tun: for gid, df_sub in g: ..
        
        A = g["A"] 
        # f(A) >> v
        # f(v, get_patient_mass(patient_ID)) >> Ek
        pass # remove again later
        # return Ek.std()
        
    # 3)
    def score_spectrum(g):
        """input df is groupby object g. Compute FFT of A. 
        Output: Mean frequency, ie (xf * yf_norm).mean() """

        SAMPLE_RATE = 51.2 # Hz
        
        # Fast Fourier transform (FFT) of acceleration A
        u = g["A"]
        
        # Eliminate static component of g, with this simple method
        u -= u.mean()  # subtract mean to each A-value (u is Series)
        
        # xf: Array of float64 
        # yf: Array of complex64
        yf = rfft(x = u.values) # compute 1-D n-point discrete Fourier Transform (DFT) of a real-valued array. (u.values is numpy.ndarray).
        xf = rfftfreq(n=len(u), d=1/SAMPLE_RATE) # calculate frequencies in center of each bin. 
        
        # Euclidean Norm of yf (complex values)
        yf_norm = np.abs(yf) # cf: https://stackoverflow.com/questions/62983674/the-absolute-value-of-a-complex-number-with-numpy
        
        return (xf * yf_norm).mean()
        


    # COMPUTE SCORES for all groups in g --> output is Series.
    scores_std = g.apply(score_std) # Series. For each group (of g), apply FUN score_std
    # scores_std = score_std(g) # Alternative zu oben!
    scores_std.name = "Standard deviation"
    
    # scores_kin = g.apply(score_kinetic_energy)
    # scores_kin.name = "Kinetic energy"

    scores_spect = g.apply(score_spectrum)
    scores_spect.name = "Characteristic MEAN frequency"
    
    # more scores
    
    # scores_all: DataFrame with MultiIndex: Patient, DeMortonDay, Side; and 2 cols: "Standard deviation" and "Characteristic MEAN frequency"
    scores_all = pd.concat([ # concat all Series
                            scores_std, 
                            # scores_kin, 
                            scores_spect
                            ], axis=1)
    
    # return scores_all
    
    # Compute means over days and sides
    # Groupby Patient, calculate mean of 2 Side * 3 Days = values
    scores = scores_all.groupby(["Patient"]).mean()
    scores["Exercise"] = ex
    scores = scores.set_index(["Exercise"], append=True) # MultiIndex with Patient & Exercise
    
    return scores
    
#  ----------------------------------------------------------------------------














