import pandas as pd
import numpy as np

# Models
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from gensim.summarization.summarizer import summarize as gensim_summarize
from gensim.summarization import keywords
from nltk.tokenize import sent_tokenize

# Metrics
from rouge import Rouge

# Language Detection
from langdetect import detect


class Summarization():
    """ The main task of this class is the implementation of different techniques for automatic summarization.
        The goal of text summarization is to create a representative abstract of the entire document, by finding
        the most informative sentences.
        The class uses 'extraction-based' methods, which work by selecting a subset of existing sentences in the
        original text to form the summary.

        At the end of the fitting procedure, the class returns a dataframe containing different columns for each
        model given as input. It's also possible to obtain a column including the keywords extracted from the text.

    Args:
    -----
        models (list or str)    >>> The list of models that we want to compute. See the results of 'getInfo_models'
                                    for the list of models that can be used.
        keywords (boolean)      >>> Optional. Set to True if you want to compute keywords. Default is set to False.

    """
    models_list = ['textrank-g', 'textrank-s', 'lexrank-s', 'lsa-s', 'luhn-s']

    def __init__(self, models):

        # check if 'models' is a list; if it isn't, we cast it.
        if type(models) != list:
            self.models = [models]
        else:
            self.models = models

        # check if each model given as input is available; if it isn't, we raise an exception.
        for model in self.models:
            if model not in Summarization.models_list:
                raise Exception("The '{}' model isn't available. Choose between {}"
                                .format(model, Summarization.models_list))

    def getInfo_models():
        print(Summarization.models_list)

    def get_lang(text):
        lang = detect(text)
        if(lang == 'it'):
            lang = 'italian'
        elif(lang == 'en'):
            lang = 'english'
        else:
            lang = ""
            print("Invalid language. The text must be Italian or English \nImpossible to recognize the following text: \n"+text)
            raise Exception("Invalid language. The text must be Italian or English")
        return lang

    def fit(self, df, field, n_sentences=1):
        """ This function fit the summarization for the models read by the class during the initialization.

            Args:
            -----
                df (pd.DataFrame)       >>> Dataframe with the text we want to summarize.

                field (str)             >>> Column name contaning the text to summarize.

                n_sentences (int)       >>> Optional. Used by sumy algorithms. Number of sentences chosen for the
                                            summary.

            Returns:
            --------
                Dataframe with all summaries obtained.

        """

        print("Summarizing...")

        df_summary = df.copy()
        for model in self.models:

            df_summary[model + '_summary'] = df[field].apply(self._summ, model=model,
                                                             n_sentences=n_sentences)

            # check if there are empty strings as models outputs. In this case, it will replace empty strings
            # with strings having length equal to 1, so that in the evaluation step we can compute scores
            # without going in error.
            string_empty = df_summary[model + '_summary'].where(df_summary[model + '_summary'] == '')\
                                                         .dropna().index.values

            for i in string_empty:
                print("Row " + str(i) + " is an empty string in " + model + "_summary")

                df_summary[model + '_summary'] = df_summary[model + '_summary']\
                    .mask(df_summary[model + '_summary'] == "", " ")

        print("Finish")
        print("")

        return df_summary

    def count_sentences(text):
        try: n_senteces = len(sent_tokenize(text, language=Summarization.get_lang(text)))
        except: n_senteces = 0
        return n_senteces

    def _summ(self, text, n_sentences, model='textrank-g'):
        """ This function creates a summary for the specified model. If no model is given, textrank with gensim
            (textrank-g) is computed by default.

            Args:
            -----
                text (string)      >>> Text that we want to summarize.

                n_sentences (int)  >>> Optional. Used by sumy algorithms. Number of sentences chosen for the summary.

                models (string)    >>> Optional. Model used to summarize.
                                       Default is set to textrank-g.

            Returns:
            --------
                String with the summary for the choosen model.
        """

        lang = Summarization.get_lang(text)

        parser = PlaintextParser.from_string(text, Tokenizer(lang))

        if (model == 'lexrank-s'):

            summarizer = LexRankSummarizer()
            summary = summarizer(parser.document, n_sentences)
            summary = [str(sentence) for sentence in summary]
            summary_p = ' '.join(summary)

        elif (model == 'textrank-s'):

            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, n_sentences)
            summary = [str(sentence) for sentence in summary]
            summary_p = ' '.join(summary)

        elif (model == 'lsa-s'):

            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, n_sentences)
            summary = [str(sentence) for sentence in summary]
            summary_p = ' '.join(summary)

        elif (model == 'luhn-s'):

            summarizer = LuhnSummarizer()
            summary = summarizer(parser.document, n_sentences)
            summary = [str(sentence) for sentence in summary]
            summary_p = ' '.join(summary)

        else:

            total_number_of_sentences = Summarization.count_sentences(text)
            ratio = n_sentences / total_number_of_sentences

            summary_p = gensim_summarize(text, ratio=ratio).replace('\n', ' ')
            return summary_p

        return summary_p

    def evaluate(self, df, field_summary, metrics='rouge-2', type_metric='f'):
        """ This function evaluate the summarization for the models used.

            Args:
            -----
                df (pandas.DataFrame)    >>> Dataframe with automatic summary and a reference summary.

                field_summary (str)      >>> Column name contaning the reference summary.

                metrics (list)           >>> Optional. List of metrics. Default is set to rouge-2.

                type_metric (str)        >>> Optional. Precision: 'p', Recall: 'r', F-score: 'f'.
                                             Default is set to 'f'.

            Returns:
            --------
                A dictionary of different dataframes for each metric computed.
                A final dataframe containing averages and standard deviation of metrics for each model.

        """
        print("Validation...")
        print("")

        if type(metrics) != list:
            metrics = [metrics]

        list_metric = ['rouge-1', 'rouge-2', 'rouge-l']

        for metric in metrics:
            if metric not in list_metric:
                raise Exception("Metric " + metric + " is not valid")

        list_typemetric = ['f', 'r', 'p']

        if type_metric not in list_typemetric:
            raise Exception("Type " + type_metric + " is not valid")

        rouge = Rouge()
        d = {}
        df_scores = pd.DataFrame()

        for metric in metrics:
            df_metric = pd.DataFrame()
            for colonna in list(df.columns):
                if colonna.find("_summary") != -1 and colonna != field_summary:
                    scores_list = []

                    for row in df.index:

                        scores = rouge.get_scores(df[colonna][row], df[field_summary][row])
                        scores_list.append(scores)

                    df_metric[colonna[:-8] + "_" + metric + "_" + type_metric] = pd.Series(
                        [x[0][metric][type_metric] for x in scores_list])

                    df_scores.loc[metric + "_" + type_metric + '_mean', colonna[:-8]] = np.mean(
                        df_metric[colonna[:-8] + "_" + metric + "_" + type_metric])

                    df_scores.loc[metric + "_" + type_metric + '_std', colonna[:-8]] = np.std(
                        df_metric[colonna[:-8] + "_" + metric + "_" + type_metric])

            d[metric + "_" + type_metric + '_df'] = df_metric
            print("Dataframe for the %s metric has been saved..." % metric)

        print("")
        print("Finish")
        return d, df_scores


class Keywords():
    '''This class create a schema to extract keyword.

        Args:
        -----
            words (int)             >>> Optional. Number of returned words.
                                        Default is set to 3.

            split (boolean)         >>> Optional. Whether split keywords if True. Defalt is set to False.

            scores (boolean)        >>> Optional. Whether score of keyword. Defalt is set to False.

            pos_filter (tuple)      >>> Optional. Part of speech filters. Defalt is set to ('NN', 'JJ').

            lemmatize (boolean)     >>> Optional. If True - lemmatize words. Default is set to False.

            deacc (boolean)         >>> Optional. If True - remove accentuation. Default is set to True.
    '''

    def __init__(self, words=3, split_keyword=False, scores=False,
                 pos_filter=('NN', 'JJ'), lemmatize=False, deacc=True):

        self.words = words
        self.split_keyword = split_keyword
        self.scores = scores
        self.pos_filter = pos_filter
        self.lemmatize = lemmatize
        self.deacc = deacc

    def fit(self, df, field):
        ''' This method extract keywords.

            Args:
            -----
                df (DataFrame)      >>> pandas.DataFrame
                field (str)         >>> Column of the text to extract keywords.

            Return:
            -------
                pandas.DataFrame with 2 columns: 'field' and 'keywords'.
        '''
        print("Keywords Extraction...")
        df_keywords = df.copy()
        df_keywords['keywords'] = df[field].apply(lambda row: self._extract_keywords(row))
        print("Finish")

        return df_keywords

    def _extract_keywords(self, text):
        """ This function extracts keywords from the text.

            Args:
            -----
                text (string)           >>> Text that we want to summarize.


            Returns:
            --------
                List of (str, float) if scores is set to True.
                List of str otherwise.
        """

        kw = keywords(text, words=self.words, split=self.split_keyword,
                      scores=self.scores, pos_filter=self.pos_filter,
                      lemmatize=self.lemmatize, deacc=self.deacc)

        if type(kw) == str:
            kw = kw.replace("\n", " ").split(" ")

        return kw
