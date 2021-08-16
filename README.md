# EEG-FeaturesExtraction

Faz a leitura dos arquivos da base de dados de EEG da universidade de Bonn.
Utilzia a biblioteca sfe (ainda não publicada) para extração de características da série tempora, do espectro de potência, do espectrograma e do biespectrograma nas bandas de frequência delta, theta, alpha, beta, gamma e do espectro inteiro.
Armazena as características em arquivos CSV nas pasta features.

### Características extraídas

##### Série Temporal

Banda     | Qtd. de características
--------- | ------
Inteiro   | 27
**Total** | **27**

##### Espectro de potência

Banda     | Qtd. de características
--------- | ------
Delta     | 29
Theta     | 29
Alpha     | 29
Beta      | 29
Gamma     | 29
Inteiro   | 29
**Total** | **174**

##### Espectrograma

Banda     | Qtd. de características
--------- | ------
Delta     | 16
Theta     | 16
Alpha     | 16
Beta      | 16
Gamma     | 16
Inteiro   | 16
**Total** | **96**

##### Biespectrograma

Banda     | Qtd. de características
--------- | ------
Delta     | 0
Theta     | 0
Alpha     | 0
Beta      | 0
Gamma     | 0
Inteiro   | 8
**Total** | **8**