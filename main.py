import pandas as pd
from Bispectrogram import Bispectrogram
from PowerSpectrum import PowerSpectrum
from Spectogram import Spectogram
from TimeSerie import TimeSerie

# ***************************************************************************************************************************
# global variables
Fs = 173.61  # frequency sample

# path of Bonn EEG dataset
path_dataset = "..\\Bases de Dados\\Universidade de Bonn\\"

folders = ['Z', 'O', 'N', 'F', 'S']

# ***************************************************************************************************************************
# Time Series Features

tsFeatures = TimeSerie(folders, path_dataset, Fs).extractAll(remove_invalids=True)
tsFeatures.to_csv('features/timeSeriesFeatures.csv', index=False, header=True, sep=';')

# ***************************************************************************************************************************
# Power Spectrum Features

pS = PowerSpectrum(folders, path_dataset, Fs)
psFAll, psFDelta, psFTheta, psFAlpha, psFBeta, psFGamma, psFEntire = pS.extract_all(remove_invalids=True)

psFAll.to_csv('features/powerSpectrumFeaturesAll.csv', index=False, header=True, sep=';')
psFDelta.to_csv('features/powerSpectrumFeaturesDelta.csv', index=False, header=True, sep=';')
psFTheta.to_csv('features/powerSpectrumFeaturesTheta.csv', index=False, header=True, sep=';')
psFAlpha.to_csv('features/powerSpectrumFeaturesAlpha.csv', index=False, header=True, sep=';')
psFBeta.to_csv('features/powerSpectrumFeaturesBeta.csv', index=False, header=True, sep=';')
psFGamma.to_csv('features/powerSpectrumFeaturesGamma.csv', index=False, header=True, sep=';')
psFEntire.to_csv('features/powerSpectrumFeaturesEntire.csv', index=False, header=True, sep=';')

# ***************************************************************************************************************************
# Spectrogram Features
sG = Spectogram(folders, path_dataset, Fs)
sgFAll, sgFDelta, sgFTheta, sgFAlpha, sgFBeta, sgFGamma, sgFEntire = sG.extractAll(remove_invalids=True)

sgFAll.to_csv('features/spectrogramFeaturesAll.csv', index=False, header=True, sep=';')
sgFDelta.to_csv('features/spectrogramFeaturesDelta.csv', index=False, header=True, sep=';')
sgFTheta.to_csv('features/spectrogramFeaturesTheta.csv', index=False, header=True, sep=';')
sgFAlpha.to_csv('features/spectrogramFeaturesAlpha.csv', index=False, header=True, sep=';')
sgFBeta.to_csv('features/spectrogramFeaturesBeta.csv', index=False, header=True, sep=';')
sgFGamma.to_csv('features/spectrogramFeaturesGamma.csv', index=False, header=True, sep=';')
sgFEntire.to_csv('features/spectrogramFeaturesEntire.csv', index=False, header=True, sep=';')

# ***************************************************************************************************************************
# Bispectrogram Features
bG = Bispectrogram(folders, path_dataset, Fs)
bgFAll, bgFDelta, bgFTheta, bgFAlpha, bgFBeta, bgFGamma, bgFEntire = bG.extractAll(remove_invalids=True)

bgFAll.to_csv('features/bispectrogramFeaturesAll.csv', index=False, header=True, sep=';')
bgFDelta.to_csv('features/bispectrogramFeaturesDelta.csv', index=False, header=True, sep=';')
bgFTheta.to_csv('features/bispectrogramFeaturesTheta.csv', index=False, header=True, sep=';')
bgFAlpha.to_csv('features/bispectrogramFeaturesAlpha.csv', index=False, header=True, sep=';')
bgFBeta.to_csv('features/bispectrogramFeaturesBeta.csv', index=False, header=True, sep=';')
bgFGamma.to_csv('features/bispectrogramFeaturesGamma.csv', index=False, header=True, sep=';')
bgFEntire.to_csv('features/bispectrogramFeaturesEntire.csv', index=False, header=True, sep=';')

# ***************************************************************************************************************************
# Todas as características já estraídas

pathAllFeatures = []
pathAllFeatures.append("features/timeSeriesFeatures.csv")
pathAllFeatures.append("features/powerSpectrumFeaturesAll.csv")
pathAllFeatures.append("features/spectrogramFeaturesAll.csv")
pathAllFeatures.append("features/bispectrogramFeaturesAll.csv")

df = pd.read_csv(pathAllFeatures[0], sep=';')
for i in range(len(pathAllFeatures)):
    if i != 0:
        df = pd.merge(df, pd.read_csv(pathAllFeatures[i], sep=';'), left_index=True, right_index=True)\
            .drop('name_y', 1).rename(columns={'name_x': 'name'})\
            .drop('class_y', 1)\
            .rename(columns={'class_x': 'class'})
# Salva o dataframe para simples verificação
df.to_csv("features/allFeatures.csv", index=False, header=True, sep=';')