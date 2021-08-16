import glob

import numpy as np
import pandas as pd

from sfe.feature_extractor_1D import ODFeatureExtractor
from sfe.sfe_aux_tools import getfgrid, spectrum_filter
from sfe.signal_transform import SignalTransform


class PowerSpectrum():
    __folders = []
    __path_dataset = ""
    __fs = 0

    # Sub-bands
    __delta_F = [1, 4]
    __theta_F = [4, 8]
    __alpha_F = [8, 12]
    __beta_F = [12, 30]
    __gamma_F = [30, 60]
    __entire_F = [1, 60]

    def __init__(self, folders, path, fs):
        self.__folders = folders
        self.__path_dataset = path
        self.__fs = fs

    def __filter_df(self, df, band):
        remove_list = list(filter(lambda col: band not in col, list(df)))
        remove_list.remove("name")
        remove_list.remove("class")
        return df.drop(remove_list, axis=1)

    def extract_all(self, remove_invalids=False):

        psAll = []

        for f in self.__folders:
            file_path = self.__path_dataset + f + "/*.txt"

            arq = glob.glob(file_path)

            for j in range(0, len(arq)):
                with open(arq[j], "r") as text_file:
                    signal = text_file.readlines()
                    signal = list(map(int, signal))

                print('Processando o espectro de potência do arquivo: ' + arq[j])

                signal = np.asarray(signal)

                stObj = SignalTransform(signal, Fs=self.__fs)

                # ------------------------------------------------------------------------------
                # power spectrum generation
                ps = stObj.get_power_spectrum()

                psFeatures = {'name': arq[j][-8:-4], 'class': f}

                # --- DELTA ---
                # power spectrum of Delta Waves
                psDelta = spectrum_filter(ps, self.__fs, self.__delta_F[0], self.__delta_F[-1])
                f_gridDelta = getfgrid(self.__fs, len(ps), fpassMin=self.__delta_F[0], fpassMax=self.__delta_F[-1])

                # power spectrum feature extraction of Delta Waves
                psDeltaO = ODFeatureExtractor(psDelta, freq_grid=f_gridDelta[:-1], label='psDelta')
                psDeltaO.extract_all_features()
                psDeltaF = psDeltaO.get_extracted_features()

                psFeatures = {**psFeatures, **psDeltaF}

                # --- THETA ---
                # power spectrum of Theta Waves
                psTheta = spectrum_filter(ps, self.__fs, self.__theta_F[0], self.__theta_F[-1])
                f_gridTheta = getfgrid(self.__fs, len(ps), fpassMin=self.__theta_F[0], fpassMax=self.__theta_F[-1])

                # power spectrum feature extraction of Theta Waves
                psThetaO = ODFeatureExtractor(psTheta, freq_grid=f_gridTheta[:-1], label='psTheta')
                psThetaO.extract_all_features()
                psThetaF = psThetaO.get_extracted_features()

                psFeatures = {**psFeatures, **psThetaF}

                # --- ALPHA ---
                # power spectrum of Alpha Waves
                psAlpha = spectrum_filter(ps, self.__fs, self.__alpha_F[0], self.__alpha_F[-1])
                f_gridAlpha = getfgrid(self.__fs, len(ps), fpassMin=self.__alpha_F[0], fpassMax=self.__alpha_F[-1])

                # power spectrum feature extraction of Alpha Waves
                psAlphaO = ODFeatureExtractor(psAlpha, freq_grid=f_gridAlpha[:-1], label='psAlpha')
                psAlphaO.extract_all_features()
                psAlphaF = psAlphaO.get_extracted_features()

                psFeatures = {**psFeatures, **psAlphaF}

                # --- BETA ---
                # power spectrum of Beta Waves
                psBeta = spectrum_filter(ps, self.__fs, self.__beta_F[0], self.__beta_F[-1])
                f_gridBeta = getfgrid(self.__fs, len(ps), fpassMin=self.__beta_F[0], fpassMax=self.__beta_F[-1])

                # power spectrum feature extraction of Beta Waves
                psBetaO = ODFeatureExtractor(psBeta, freq_grid=f_gridBeta[:-1], label='psBeta')
                psBetaO.extract_all_features()
                psBetaF = psBetaO.get_extracted_features()

                psFeatures = {**psFeatures, **psBetaF}

                # --- GAMMA ---
                # power spectrum of Gamma Waves
                psGamma = spectrum_filter(ps, self.__fs, self.__gamma_F[0], self.__gamma_F[-1])
                f_gridGamma = getfgrid(self.__fs, len(ps), fpassMin=self.__gamma_F[0], fpassMax=self.__gamma_F[-1])

                # power spectrum feature extraction of Gamma Waves
                psGammaO = ODFeatureExtractor(psGamma, freq_grid=f_gridGamma[:-1], label='psGamma')
                psGammaO.extract_all_features()
                psGammaF = psGammaO.get_extracted_features()

                psFeatures = {**psFeatures, **psGammaF}

                # --- ENTIRE ---
                # power spectrum of Total Waves
                psEmtire = spectrum_filter(ps, self.__fs, self.__entire_F[0], self.__entire_F[-1])
                f_gridEntire = getfgrid(self.__fs, len(ps), fpassMin=self.__entire_F[0], fpassMax=self.__entire_F[-1])

                # power spectrum feature extraction of Gamma Waves
                psEntirelO = ODFeatureExtractor(psEmtire, freq_grid=f_gridEntire[:-1], label='psEntire')
                psEntirelO.extract_all_features()
                psEntireF = psEntirelO.get_extracted_features()

                psFeatures = {**psFeatures, **psEntireF}

                psAll.append(psFeatures)

        # Converte o array em um dataframe com todas as características
        _dfPsAll = pd.DataFrame(psAll)

        if remove_invalids:
            # Remove todas as colunas que têm inf -inf e NaN
            _dfPsAll = _dfPsAll.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        _dfPsDelta = self.__filter_df(_dfPsAll, "psDelta")
        _dfPsTheta = self.__filter_df(_dfPsAll, "psTheta")
        _dfPsAlpha = self.__filter_df(_dfPsAll, "psAlpha")
        _dfPsBeta = self.__filter_df(_dfPsAll, "psBeta")
        _dfPsGamma = self.__filter_df(_dfPsAll, "psGamma")
        _dfPsEntire = self.__filter_df(_dfPsAll, "psEntire")

        return _dfPsAll, _dfPsDelta, _dfPsTheta, _dfPsAlpha, _dfPsBeta, _dfPsGamma, _dfPsEntire
