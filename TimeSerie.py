import glob

import numpy as np
import pandas as pd

# Usar se a biblioteca não estiver no mesmo diretório do projeto. O autocomplete não irá funcionar.
# sys.path.append(os.path.abspath("D:\\Repositorios\\signal_feature_extraction"))
from sfe.feature_extractor_1D import ODFeatureExtractor
from sfe.sfe_aux_tools import getfgrid


class TimeSerie():
    __folders = []
    __path_dataset = ""
    __fs = 0

    def __init__(self, folders, path, fs):
        self.__folders = folders
        self.__path_dataset = path
        self.__fs = fs

    def extractAll(self, remove_invalids=False):

        _tsAll = []

        for f in self.__folders:
            file_path = self.__path_dataset + f + "/*.txt"

            arq = glob.glob(file_path)

            for j in range(0, len(arq)):
                with open(arq[j], "r") as text_file:
                    signal = text_file.readlines()
                    signal = list(map(int, signal))

                print('Processando a série temporal do arquivo: ' + arq[j])

                signal = np.asarray(signal)

                f_grid = getfgrid(self.__fs, len(signal))
                tsO = ODFeatureExtractor(signal, freq_grid=f_grid, label='ts')
                tsO.extract_all_features()
                tsFeatures = {'name': arq[j][-8:-4], 'class': f}
                tsFeatures = {**tsFeatures, **tsO.get_extracted_features()}

                _tsAll.append(tsFeatures)

        _dfTsAll = pd.DataFrame(_tsAll)

        if remove_invalids:
            # Remove todas as colunas que têm inf -inf e NaN
            _dfTsAll = _dfTsAll.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        return _dfTsAll
