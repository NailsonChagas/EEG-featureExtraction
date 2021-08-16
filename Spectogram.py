import glob

import numpy as np
import pandas as pd

from sfe.SpectrogramFeatures import SpectrogramFeatures
from sfe.signal_transform import SignalTransform


class Spectogram():
    __folders = []
    __path_dataset = ""
    __fs = 0

    def __init__(self, folders, path, fs):
        self.__folders = folders
        self.__path_dataset = path
        self.__fs = fs

    def __filter_df(self, df, band):
        remove_list = list(filter(lambda col: band not in col, list(df)))
        remove_list.remove("name")
        remove_list.remove("class")
        return df.drop(remove_list, axis=1)

    def extractAll(self, remove_invalids=False):

        spAll = []

        for f in self.__folders:
            file_path = self.__path_dataset + f + "/*.txt"

            arq = glob.glob(file_path)

            for j in range(0, len(arq)):
                with open(arq[j], "r") as text_file:
                    signal = text_file.readlines()
                    signal = list(map(int, signal))

                print('Processando o espectograma do arquivo: ' + arq[j])

                signal = np.asarray(signal)

                stObj = SignalTransform(signal, Fs=self.__fs)

                # ------------------------------------------------------------------------------
                # spectogram generation
                sg = stObj.get_spectrogram()

                sgFeatures = {'name': arq[j][-8:-4], 'class': f}

                sgSignal = SpectrogramFeatures(sg, self.__fs, len(sg))
                sgFeatures = {**sgFeatures, **sgSignal.extract_features()}
                spAll.append(sgFeatures)

        # Converte o array em um dataframe com todas as características
        _dfSgAll = pd.DataFrame(spAll)

        if remove_invalids:
            # Remove todas as colunas que têm inf -inf e NaN
            _dfSgAll = _dfSgAll.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        _dfSgDelta = self.__filter_df(_dfSgAll, "delta")
        _dfSgTheta = self.__filter_df(_dfSgAll, "theta")
        _dfSgAlpha = self.__filter_df(_dfSgAll, "alpha")
        _dfSgBeta = self.__filter_df(_dfSgAll, "beta")
        _dfSgGamma = self.__filter_df(_dfSgAll, "gamma")
        _dfSgEntire = self.__filter_df(_dfSgAll, "entire")

        return _dfSgAll, _dfSgDelta, _dfSgTheta, _dfSgAlpha, _dfSgBeta, _dfSgGamma, _dfSgEntire
