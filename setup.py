from setuptools import setup
from setuptools import find_packages

setup(name='mltools',
      version='0.1',
      description='A custom collection of ML methods and utilities',
      url='https://github.com/Tostox/mltools',
      author='Gabriele Sotto',
      license='GPLv3',
      install_requires=['numpy>=1.15.2', 'pandas>=0.23.4', 'scikit-learn >= 0.19.1',
                        'scipy>=1.1.0', 'hyperopt>=0.1',
                        'missingno>=0.4.1', 'matplotlib>=2.2.2', 'seaborn>=0.9.0',
                        'deap>=1.2.2', 'lightgbm>=2.0.6', 'xgboost>=0.80', 'bokeh>=0.13.0',
                        'gensim>=3.4.0', 'nltk>=3.4', 'sumy>=0.7.0', 'rouge>=0.3.1',
                        'langdetect>=1.0.7', 'spacy>=2.0.11', 'statsmodels>=0.9.0',
                        'fbprophet>=0.3.post2', 'googletrans>=2.4.0'],
      packages=find_packages(),
      zip_safe=False)
