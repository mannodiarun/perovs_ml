This notebook reads a DFT dataset of ABX3 perovskites and trains ML models for predicting their band gaps (PBE and HSE), refractive index, and photovoltaic figure of merit.

![image](https://user-images.githubusercontent.com/44070997/121405240-ccfeeb80-c92a-11eb-8719-9c7dc731eb00.png)


In this notebook, we: 
(a) read the data, including DFT computed properties and descriptors
(b) initize neural network frameworks for regression using keras
(c) train and test best regression models, visualize the predictions
