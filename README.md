# mltools
This repository include my personal custom python package for ML-Data Science tasks.

## Description 

The package *mltools* is an high-level implementation of various packages and modules for Machine Learning and Data Mining. The idea behind this work is to create a python package that allow an easy and fast usage of the typical algorithms used by Data Scientist, in order to build pipelines that are very easy to deploy. *mltools* is a collection of custom methods and custom re-implementations of functions and tools that are coming from python packages like *sk-learn*, *scipy*, *nltk*, *gensim* etc.
The package include different modules that are specific for a different fields of analysis. In version 1.0 we have four main modules:
* evaluateModels: module for train and evaluate the main ML algorithms for classification and regression task; 
* featureEngineering: module for data preprocessing and exploration step, include various methods for feature selection;
* textMining: module for text analytics. Include functions for text cleaning, vectorization, summarization and data augmentation;
* timeSeriesTools: module for time series analytics. Provide methods for analize temporal data such as smoothing filter, ETS analysis, forecasting, change point detector etc.

## Installation

```bash
pip install git+https://github.com/Tostox/mltools.git
```

## Requirements

* Python >= 3.6
* numpy >= 1.15.2
* pandas >= 0.23.4
* sklearn >= 0.19.1
* sklearn-genetic >= 0.1
* scipy >= 1.1.0
* hyperopt >= 0.1
* missingno >= 0.4.1
* matplotlib >= 2.2.2
* seaborn >= 0.9.0
* deap >= 1.2.2
* lightgbm >= 2.0.6
* xgboost >= 0.80
* bokeh >= 0.13.0
* gensim >= 3.4.0
* nltk >= 3.4
* sumy >= 0.7.0
* rouge >= 0.3.1
* langdetect >= 1.0.7
* spacy >= 2.0.11
* statsmodels >= 0.9.0
* fbprophet >= 0.3.post2
* googletrans >= 2.4.0

## Acknowledgments

A special thanks to Martina Trojani and Francesca Casini for the collaboration and to Manuel Calzolari (https://github.com/manuel-calzolari) that provided me the code for the genetic feature selection module.
