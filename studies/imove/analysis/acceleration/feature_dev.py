# FEATURE DEVELOPMENT

# LIBRARIES ----------------------------------------------------------------------------
import pandas as pd
import numpy as np

# Import own modules
from acceleration import align_timestamp


# FEATURE DEVELOPMENT ----------------------------------------------------------------------------

# Und dann noch das Rezept für die für die Berechnung der Features.
# Angenommen, wir möchten verschiedene Features für verschiedene Übungen bestimmen:

# Selektiere die relevanten Daten (ohne Margin) für eine bestimmte üung
    # Gruppiere die Daten nach Patienten, nach Seite, nach Tag
    # Für die Data-Chunks, rufe Score-Funktionen auf, die du vorher definiert hast
    # Sammle den Output
    # Berechne Mittelwerte
    # Berechne Statistik über verschiedene Patienten
def XX(df, ex='12'):
    """Subset input df  """
    
    # subset input df (without margins) for specific ex 
    mask = df.DeMortonDay.eq(day)
    df = df[mask]
        
    
    
    
    
# Compute some score
def score_std(df):
    return df["A"].std()

def score_kinetic_energy(df, masses):
    # Compute kinetic energy given acceleration and mass
    # https://en.wikipedia.org/wiki/Kinetic_energy
    # Argument masses: a lookup patient -> body mass
    # I thought this information is available somewhere...
    pat = df["Patient"].iloc[0]
    mass = masses[pat]
    ...
def score_spectrum(df):
    # ...
    
# Example: Compute a score for exercise 12
data = df.loc[df["DeMortonLabel"]=="12"]
g = data.groupby(["Patient", "DeMortonDay", "Side"])
scores_std = g.apply(score_std)   # Same as g["A"].std()
scores_std.name = "Standard deviation"
scores_kin = g.apply(score_kinetic_energy, masses)
scores_kin.name = "Kinetic energy"
scores_spect = g.apply(score_spectrum)
scores_spect.name = "Characteristic frequency"

# more scores...
scores_all = pd.concat([scores_std, scores_kin, ...], axis=1)

# Compute means over days and sides
scores = scores_all.groupby(["Patient"]).mean()