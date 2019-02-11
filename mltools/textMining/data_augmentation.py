import emoji
import random
import re
from googletrans import Translator
import numpy as np
import pandas as pd



class DataAugmentation():
    """Class to perform a data augmentation for text mining task.
        The methods avaible are:
            1. backtranslation
            2. rand_replacement: random words replacement (NOT IMPLEMENTED YET)

    Args:
    -----
        method (string) >>> strings that specify the method that we want to use
        for the data augmentation. By default is set to backtranslation.
        args (dict) >>> dictionary of keywords arguments to pass to the method invoke

    """
     
    def __init__(self, method, args = None):
        
        self.method=method
        self.args = args

    def fit(self, data):
        """
        Args:
        -----
            data (pandas.Series) >>> text string to process.

        Returns:
        --------
            pandas.Series that contain the original text "back-translated"
        """

        if self.method :
            print("Back-Translation...")
            augmented_data = data.apply(self._backtranslation)
            print("done")

        return augmented_data
            
    def _backtranslation(self, text, src = 'en', dest = 'zh-tw'):
        
        if self.args is not None:
            try:
                src = self.args['src']
                dest = self.args['dest']
            except: raise Exception('Invalid argument. Try to pass src and dest')
        
        translator = Translator()
        try:
            translated = translator.translate(str(text), src=src, dest=dest).text # Pass both source and destination
            translated = translator.translate(str(translated), src=dest, dest=src).text # Pass both source and destination
        except:
            translated=""

        del translator
        return translated
