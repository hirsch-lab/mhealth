#### LIBRARIES ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

#### FOURIER TRANSFORMATION ----------------------------------------------------------------------------


# df_chunk: Data from one patient, one exercise, one day, one side
# df_chunk = ...
# freq = 51       
    
# Sample rate of the data.                    
# Could be computed also from data:                    
# timestamps = df_chunk.index                   
# dt = np.diff(timestamps).mean()                   
# freq = 1/.total_seconds()                    
# (Note: why 51Hz? I thought it was 60Hz...)


# yf = rfft(u.values) # Discrete Fourier transform (for real input)
# xf = rfftfreq(len(u), 1 / freq)

# Plot the amplitude spectrum
# plt.plot(xf, np.abs(yf))
# plt.show()

# Kommentar: Man könnte den "statischen Anteil" auch anders berechnen 
# (d.h. der Anteil, der vom Schwerefeld der Erde verursacht wird) als u.mean(). 
# Aber es ist mal ein einfacher Ansatz.


def fourier_transform(df, pat='002', ex='12', day='2', side='left'):
    """Subset input df (eg df_aligned) and run FFT. Outputs: xf, yf. """
    
    # subset inputed df 
    mask = df.Patient.eq(pat) & df.DeMortonLabel.eq(ex) & df.DeMortonDay.eq(day) & df.Side.eq(side)
    df = df[mask]
    
    # calculate duration (= time between start and end of signal)
    timestamps = df.index
    dt = np.diff(timestamps).mean() # mean Zeitabstand bw 2 data points [sec]
    duration = dt * len(timestamps) # 1*6
    print("duration of signal is: ", duration) # 6

    SAMPLE_RATE = 51.2  # In Hz. Equivalent to freq (frequency of acc data). 
    SAMPLE_RATE = 1

    # SAMPLE_RATE determines how many data points the signal uses to represent the sine wave per second
    # alternatively, calculate freq from data..
    # timestamps = df_chunk.index
    # dt = np.diff(timestamps).mean() # Calculate differences bw rows. Mean Zeitabstand bw data points.
    # 1/timestamps.total_seconds()
    
    N = SAMPLE_RATE * duration # number of samples. # 1*6
    N = int(N)
    
    # Fast Fourier transform (FFT) of acceleration A
    u = df["A"]
    
    # Eliminate static component of g, with this simple method
    u -= u.mean()  # subtract mean to each A-value (u is Series)
    
    yf = rfft(x = u.values) # compute 1-D n-point discrete Fourier Transform (DFT) of a real-valued array. (u.values is numpy.ndarray).
    # if length of x is even: output is  (n/2)+1. 
    # if length of x is odd : output is  (n+1)/2. 
    yf[0]; yf[1];yf[2];yf[3]
    
    # rfftfreq(): calculate frequencies in center of each bin. 
    xf = rfftfreq(n=N, d=1/SAMPLE_RATE) # n: window length. d: sample spacing (inverse of sample rate)
    # xf = rfftfreq(n=len(u), d=1/SAMPLE_RATE) # alternative
    
    return xf, yf









