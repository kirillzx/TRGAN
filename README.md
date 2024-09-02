# TRGAN: A Time-Dependent Generative Adversarial Network for Synthetic Transactional Data Generation
### Kirill Zakharov, Elizaveta Stavinova, Anton Lysenko

The official realisation of the prposed method TRGAN

Cite:

@inproceedings{10.1145/3641067.3641076,
author = {Zakharov, Kirill and Stavinova, Elizaveta and Lysenko, Anton},
title = {TRGAN: A Time-Dependent Generative Adversarial Network for Synthetic Transactional Data Generation},
year = {2024},
isbn = {9798400717239},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3641067.3641076},
doi = {10.1145/3641067.3641076},
booktitle = {Proceedings of the 2023 7th International Conference on Software and E-Business},
pages = {1–8},
numpages = {8},
location = {<conf-loc>, <city>Osaka</city>, <country>Japan</country>, </conf-loc>},
series = {ICSeB '23}
}

Kirill Zakharov, Elizaveta Stavinova, and Anton Lysenko. 2023. TRGAN:
A Time-Dependent Generative Adversarial Network for Synthetic Trans-
actional Data Generation. In 2023 7th International Conference on Software
and e-Business (ICSeB 2023), December 21–23, 2023, Osaka, Japan. ACM, New
York, NY, USA, 8 pages. https://doi.org/10.1145/3641067.3641076


We have proposed a new approach for synthetic bank transaction generation with time factor. For that we developed the mechanism for synthetic time generation based on Poisson processes, preprocessing scheme for all kinds of attributes in transactional data and the new GAN architecture with generator, supervisor and two discriminators with conditional vector depending on time.

## Data
All data and pretrained models from the last updates can be found by the link on google drive https://drive.google.com/drive/folders/1mw3uUlw2yGz6N6BiaPe-5G-Iv-4s-6EO?usp=sharing.

## Method

Open the TRGAN_V2_train.ipynb file.

The general pipeline includes the following architecture: generator, supervisor and two discriminators. As Input we use a bounded stochastic process from the DCL family instead of standard Gaussian Noise for time dependencies and smoothenes of transitions between transactions. For a more detailed description see the article the section Generative Method.

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/general.png"  width="60%" height="30%">
</p>

We implement the new preprocessing scheme which work for categorical attributes with a big number of unique values (Frequeny Encoder plus DP Autoencoder), categorical attributes with a small number of unique values (OneHot plus DP Autoencoder), numerical attributes (Gaussian Normalize, MinMax plus DP Autoencoder) and time attribute (sine and cosine values for days and months).

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/preprocessing.png"  width="60%" height="30%">
</p>

Our approach TRGAN use conditional generation with time-dependent factor. Therefore we use conditional vector with time factor and the information about categories of clients in a specific time period. 

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/cv.png"  width="60%" height="30%">
</p>

## Hyperparameters
Were have prepared the table of hyperparameters which were used for experiments in the article. Every parameters can be changed in the follwong files: TRGAN_main.py and TRGAN notebooks .ipynb for each dataset.

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/hyper.png"  width="60%" height="30%">
</p>


## Results
Comparison of synthetic amount and the real one. 

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/amount.png"  width="60%" height="30%">
</p>

Comparison of synthetic categories and the real one. 
<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/mcc.png"  width="60%" height="30%">
</p>

Comparison of synthetic deltas and the real one (Synthetic time generation mechanism). 
<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/delta.png"  width="60%" height="30%">
</p>

## Scenario modelling

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/scenario_by_mcc.png"  width="60%" height="30%">
</p>

<p align="center">
<img src="https://github.com/kirillzx/TRGAN/blob/main/Images/scenario_by_value.png"  width="60%" height="30%">
</p>
