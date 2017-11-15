# SiMpLe
## CS 273A - Machine Learning course project

###### This project is for an in-class Kaggle competition at UC Irvine. For this competition, we predict whether there is rainfall at a location, based on (processed) infrared satellite image information. 

###### The competition data are satellite-based measurements of cloud temperature (infrared imaging), being used to predict the presence or absence of rainfall at a particular location. The data are courtesy of the [UC Irvine Center for Hydrometeorology and Remote Sensing](http://chrs.web.uci.edu/), and have been pre-processed to extract features corresponding to a model they use actively for predicting rainfall across the globe. Each data point corresponds to a particular lat-long location where the model thinks there might be rain; the extracted features include information such as IR temperature at that location, and information about the corresponding cloud (area, average temperature, etc.). The target value is a binary indicator of whether there was rain (measured by radar) at that location; you will notice that the data are slightly imbalanced (positives make up about 30% of the training data).
