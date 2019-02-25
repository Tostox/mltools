import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import functools
import spacy


class TextPreprocessing():
    """ This class the preprocessing of a documents (text field) that are contained into a pandas DataFrame.
        It's possible to perform the following step:
            1. Text standardization with data cleaning (removal of special characters, numbers, link etc.).
            2. Removal of stopwords.
            3. Lemmatization.

    Args:
    -----
        lemmatization (boolean) >>> Set it to True if you want to perform the lemmatization step.
                                    Defalt is equal to False.
        standardize (boolean)   >>> Set it to True if you want to perform the standardization step.
                                    Defalt is equal to True.
        chr_to_remove (list)    >>> List of strings (regex) that represent the vector of special char or string that
                                    you want to remove.
        chr_to_keep (regex)     >>> regex that represent the char that you want to keep. By default this method return
                                    only the letters of english alphabet. If you want to keep another special character
                                    you can specify this by setting this field.
        language  (string)      >>> Optional. Language of the text that you want to analyze.
                                    Default is 'en' (English).
    """

    def __init__(self, lemmatization=False,
                 standardize=True,
                 remove_stopwords=True,
                 chr_to_remove=[r"http\S+", r"http", r"@\S+", r"@", r""],
                 chr_to_keep=r"[^A-Za-z]",
                 language='en'):

        self.lemmatization = lemmatization
        self.standardize = standardize
        self.chr_to_remove = chr_to_remove
        self.chr_to_keep = chr_to_keep
        self.language = language

        # Stopwords
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            if self.language == 'en':
                self.stoplist = stopwords.words('english')
            elif self.language == 'it':
                self.stoplist = stopwords.words('italian')
            else:
                raise Exception("Invalid language")
        else:
            self.stoplist = []

    def is_null(self, text):
        return text.isspace()

    def standardize_text(self, df, text_field):

        for regexp in self.chr_to_remove:
            df[text_field] = df[text_field].str.replace(regexp, "")

        df[text_field] = df[text_field].str.replace(self.chr_to_keep, " ")  # we not consider numbers
        df[text_field] = df[text_field].str.lower()
        return df

    def rm_stopwords(self, text):
        clearlist = [word for word in text if word not in self.stoplist]
        return clearlist

    def lemmatizer(self, text):
        if self.language == 'it':
            nlp = spacy.load('it_core_news_sm')
        else:
            nlp = spacy.load(self.language)
        sent = []
        doc = nlp(" ".join(text))
        for word in doc:
            sent.append(word.lemma_)
        return sent

    def tokenization(self, text, min_len):
        tokenizer = RegexpTokenizer(r'\w+')
        token_list = tokenizer.tokenize(text)
        token_list = [token for token in token_list if len(token) > min_len]
        token_list = [token for token in token_list if token not in self.stoplist]
        return token_list

    def fit(self, data_df, field, min_len=3):
        """
        Args:
        -----
            data_df (pandas.DataFrame) >>> dataframe that contains the documents and the text field to process.
            field (string) >>> name of the field (column) that contain the text to process.
            min_len (int) >>> minimum length of the word

        Returns:
        --------
            pandas.DataFrame that are the copy of the original dataframe plus a column that contain the clean tokens
            ("tokens") and (if computed) another field with the lemma of these tokens ("lemma").
        """

        # Drop fields which contain only space char
        print("Data cleaning...")
        df = data_df[~data_df[field].apply(self.is_null)]

        # Standardization
        if self.standardize:
            print("Standardization...")
            df = self.standardize_text(df, field)

        # Token extraction
        print("Tokenization...")
        df["tokens"] = df[field].apply(self.tokenization, min_len=min_len)

        # Stopwords
        if self.remove_stopwords:
            print("Removing stopwords...")
            df["tokens"] = df["tokens"].apply(self.rm_stopwords)

        # Lemmatization
        if self.lemmatization:
            print("Lemmatization...")
            df["lemma"] = df["tokens"].apply(self.lemmatizer)
        print("Finish")

        return df


class VectorizeData():

    """ This class compute the vectorization of a list of text tokens.
    Args:
    -----
        method (string) >>> The metric used to transform the feature data. The choices are "binary", "count", "tf" or "tf-idf".
                            Default is "tf-idf".

    """

    def __init__(self, method='tf-idf'):
        self.method = method

    def fit(self, train_data, test_data = None, skipper = 'word', max_features = 60000):
        """
        Args:
        -----
            train_data (pandas.Series) >>> column of the training set dataframe that contains the tokens to process.
            test_data (pandas.Series) >>> column of the test set dataframe that contains the tokens to process.

        Returns:
        --------
            a tuple contains the two feature matrix (training data and test data).
        """

        if self.method == 'binary':
            count_vectorizer = TfidfVectorizer(analyzer = skipper, binary=True, use_idf=False, norm=None, max_features=max_features)
        elif self.method == 'count':
            count_vectorizer = CountVectorizer(analyzer = skipper)
        elif self.method == 'tf':
            count_vectorizer = TfidfVectorizer(analyzer = skipper, use_idf=False, max_features=max_features)
        elif self.method == 'tf-idf':
            count_vectorizer = TfidfVectorizer(analyzer = skipper, max_features=max_features)
        else:
            raise Exception("Invalid method. Use: binary, tf or tf-idf")
        
        if skipper == 'word':
            x_train = train_data.apply(str)
        else:
            x_train = train_data

        transformed_train_data = count_vectorizer.fit_transform(x_train)
        if test_data is not None:
            if skipper == 'word':
                x_test = test_data.apply(str)
            else:
                x_test = test_data
            transformed_test_data = count_vectorizer.transform(x_test)
            return transformed_train_data, transformed_test_data, count_vectorizer
        else:
            return transformed_train_data, count_vectorizer


def get_n_skipgrams(tokens, n, k):
    """
        Fuction to extract the skip gram of a given list of tokens
        Args:
        -----
        tokens (list of strings) >>> list of tokens extracted by using the TextPreprocessing class.
        n (int) >>> number of gram.
        k (int) >>> skip param for the skip gram.
        
        Returns:
        --------
        a list of tuple that contains the skip gram.
        """
    
    sent = " ".join(tokens).split()
    return list(nltk.skipgrams(sent, n, k))
