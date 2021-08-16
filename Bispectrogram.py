import glob

import numpy as np
import pandas as pd

from sfe.BispectrogramFeatures import BispectrogramFeatures
from sfe.signal_transform import SignalTransform


class Bispectrogram():
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

        bgAll = []

        for f in self.__folders:
            file_path = self.__path_dataset + f + "/*.txt"

            arq = glob.glob(file_path)

            for j in range(0, len(arq)):
                with open(arq[j], "r") as text_file:
                    signal = text_file.readlines()
                    signal = list(map(int, signal))

                print('Processando o biespectograma do arquivo: ' + arq[j])

                signal = np.asarray(signal)

                stObj = SignalTransform(signal, Fs=self.__fs)

                # ------------------------------------------------------------------------------
                # spectogram generation
                bg = stObj.get_bispectrogram()

                bgFeatures = {'name': arq[j][-8:-4], 'class': f}

                bgSignal = BispectrogramFeatures(bg, self.__fs, len(bg))
                bgFeatures = {**bgFeatures, **bgSignal.extract_features()}
                bgAll.append(bgFeatures)

        # Converte o array em um dataframe com todas as características
        _dfBgAll = pd.DataFrame(bgAll)

        if remove_invalids:
            # Remove todas as colunas que têm inf -inf e NaN
            _dfBgAll = _dfBgAll.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        _dfBgDelta = self.__filter_df(_dfBgAll, "delta")
        _dfBgTheta = self.__filter_df(_dfBgAll, "theta")
        _dfBgAlpha = self.__filter_df(_dfBgAll, "alpha")
        _dfBgBeta = self.__filter_df(_dfBgAll, "beta")
        _dfBgGamma = self.__filter_df(_dfBgAll, "gamma")
        _dfBgEntire = self.__filter_df(_dfBgAll, "entire")

        return _dfBgAll, _dfBgDelta, _dfBgTheta, _dfBgAlpha, _dfBgBeta, _dfBgGamma, _dfBgEntire
