# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq


# Import own modules
from acceleration import resample, align_timestamp
from paths import path_data


def get_patient_mass(lookup, patient_ID):
    """for specific patient_ID, return mass, ie weight."""
    return lookup.loc[patient_ID, "weight"]


def get_patient_bmi(lookup, patient_ID):
    """for specific patient_ID, return bmi."""
    return lookup.loc[patient_ID, "BMI"]


def get_patient_borg_exertion(patient_ID, day):
    """for specific patient_ID, day, return exertion."""
    # exertion = df_borg.copy()  # Don't modify the original file.
    exertion = pd.read_csv(Path(path_data, 'Borg/exertion.csv') ) # load exertion.csv which was created in R
    exertion.loc[:,'patient_ID'] = exertion.patient_ID.apply(str) # int -> str
    exertion.loc[:,'patient_ID'] = exertion['patient_ID'].str.rjust(3, "0") # add trailing zer
    exertion['day'] = exertion['day'].map(lambda x: x.lstrip('D')) # D1 --> 1
    exertion = exertion.set_index(['patient_ID', 'day'])
    exertion_value = exertion.loc[(patient_ID, day), "exertion"] # extract exertion value
    return exertion_value


# a ----------------------------------------------------------------------------
def feature_development(df, df_borg, ex='12'): # df und df_borg are passed to sub-functions.
    """Index of df: RangeIndex
    """
    # subset input df (without margins) for specific ex
    df = df[df.DeMortonLabel == ex]

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
        scores_std.name = "Std Dev of A"

        return scores_std

    # 2)
    def score_bmi(df, lookup):
        """input df is groupby object g. Return bmi of Patient per group."""
        df = align_timestamp(df=df, grouping=['Patient', 'DeMortonDay', 'Side'])

        def transform(df, lookup): # lookup=df_borg
            """to be applied on each subgroup. Calculate Standard deviation of A."""
            assert df.Patient.nunique() == 1
            patient_ID = df.Patient.iloc[0]  # Take first element
            bmi = get_patient_bmi(lookup=lookup, patient_ID=patient_ID)
            return bmi

        # groupby Patient, Side, DeMortonDay
        g = df.groupby(["Patient", "DeMortonDay", "Side"])

        scores_bmi = g.apply(transform, lookup=lookup)
        scores_bmi.name = "BMI"

        return scores_bmi


    # 3)
    def score_kinetic_energy(df, lookup, method='2'): # method 2 fuses Sensors 'left' and 'right'
        """input df is groupby object g.
        Compute kinetic energy given acceleration and mass"""

        df = align_timestamp(df=df, grouping=['Patient', 'DeMortonDay', 'Side'])

        # Make col 'time' the new Index (Int64Index is no more index)
        df = df.set_index('time') # needed for resample() later

        def transform(df, lookup): # warum müssen wir resamplen?
            """to be applied on each subgroup. resample A. Calculate E_kin_total.
            f(A) >> v
            f(v, get_patient_mass(patient_ID)) >> Ek """

            dt = 1 # 0.5 # Time step in seconds
            v0 = 0 #  assumption for velocity at time 0
            rule = "1s"

            # get mass of Patient_ID (normal col)
            patient_ID = df.Patient.iloc[0] # Get 1st element of "003", "003", "003", "003",
            m = get_patient_mass(lookup=lookup, patient_ID=patient_ID)

            ## Method 1: Per patient, day, exercise and side
            if method=='1':
                A = df["A"]
                A_filt = A.resample(rule=rule).mean() # Ist das korrekt?

            ## Method 2: Per patient, day, exercise. But NOT per side.
            elif method=='2':
                A_left  = df.loc[df["Side"]=="left" , "A"]
                A_right = df.loc[df["Side"]=="right", "A"]
                A_left  = A_left.resample(rule=rule).mean()
                A_right = A_right.resample(rule=rule).mean()
                A_filt  = 0.5*(A_left+A_right)

            else:
                print("There is an error with the method")
                pass

            # Remove the static component. This is plain wrong, but better
            # than doing nothing. What we would have to do is:
            # - Estimate the direction of the gravitational acceleration.
            # - Remove the gravitational component from the acceleration VECTOR:
            #   a_vec = a_vec - g_vec
            # - Then: Compute the amplitude: np.sqrt((a_vec**2).sum())
            #   Or: estimate the "horizontal" acceleration (normal to the
            #   estimated direction of gravitation.
            #   However: It's a bit tricky to do this.
            A_filt -= A_filt.median() # subtract median from each value

            v_filt = A_filt.cumsum()*dt + v0 # cumsum() beginnt bei 0 implizit
            E_kin  = m*v_filt**2 # variable m inputed
            E_kin_total = E_kin.sum() # aufsummieren

            return E_kin_total

        ## GROUPBY
        if method=='1': # groupby: Patient, DeMortonDay, Side
            groupby = ["Patient", "DeMortonDay", "Side"]
            g = df.groupby(groupby)
            scores_kin = g.apply(transform, lookup=lookup)
        elif method=='2':
            # groupby: Patient, DeMortonDay. But NOT per Side.
            groupby = ["Patient", "DeMortonDay"]
            g = df.groupby(groupby)
            scores_kin = g.apply(transform, lookup=lookup)
            scores_kin = pd.concat([scores_kin, scores_kin],  # verstehe logik nicht
                                   keys=["left", "right"], axis=0) # concat on top of each other
            scores_kin.index.set_names("Side", level=0, inplace=True)
            scores_kin = scores_kin.reorder_levels([1,2,0])
        else:
            print("There is an error with the method")
            pass

        scores_kin.name = "Kinetic energy"
        return scores_kin


    # 4)
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
        scores_kin.name = "Max. Frequency"
        return scores_kin

    # 5)
    def score_borg_exertion(df):
        """input df is groupby object g. Compute borg_exertion of Pat."""
        df = align_timestamp(df=df, grouping=['Patient', 'DeMortonDay', 'Side']) # grouping nach Side macht eig. keinen Sinn

        def transform(df):
            """to be applied on each subgroup."""
            patient_ID = df.Patient.iloc[0] # Get 1st element of "003", "003", "003", "003",
            day = df.DeMortonDay.iloc[0] # Get 1st element of 2, 2, 2,
            score_exertion = get_patient_borg_exertion(patient_ID, day) # LUT
            return score_exertion #

        # groupby Patient, Side, DeMortonDay
        g = df.groupby(["Patient", "DeMortonDay", "Side"])

        scores_exertion = g.apply(transform)
        scores_exertion.name = "Exertion"

        return scores_exertion


    ## Execute all inner functions
    scores_std   = score_std(df)
    scores_bmi   = score_bmi(df, lookup=df_borg)
    scores_kin   = score_kinetic_energy(df, lookup=df_borg, method='2') # method 2 fuses left and right sensors
    scores_spect = score_spectrum(df)
    scores_exertion = score_borg_exertion(df) # exertion.csv is loaded within the function.

    # scores_all: DataFrame with MultiIndex: Patient, DeMortonDay, Side.
    scores_all = pd.concat([ # concat all Series
                            scores_std,
                            scores_bmi,
                            scores_kin,
                            scores_spect,
                            scores_exertion
                            ], axis=1)

    scores_all["Exercise"] = ex
    scores_all = scores_all.set_index("Exercise", append=True) # extend MultiIndex with "Exercise"

    return scores_all



# # exercises = ['2a', '5a','12','15']
# fd_2a = feature_development(df=acc, ex='2a')
# fd_5a = feature_development(df=acc, ex='5a')

# fd_12 = feature_development(df=acc, ex='12')
# fd_15 = feature_development(df=acc, ex='15')

# scores_ALL_ex = pd.DataFrame()
# scores_ALL_ex = scores_ALL_ex.append(fd_12)
# scores_ALL_ex = scores_ALL_ex.append(fd_15)
# scores_ALL_ex = scores_ALL_ex.append(fd_2a)
# scores_ALL_ex = scores_ALL_ex.append(fd_5a)

# df = acc
# mask = df.DeMortonLabel.eq('12')
# df = df[mask]
