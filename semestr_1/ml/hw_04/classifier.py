import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import everygrams
from typing import List, Iterator, Tuple, Union, Generator
import joblib
import warnings
from scipy.special import softmax


class Tokenizer:
    def __init__(self) -> None:
        self.orig_text_series = None
        self.token_series = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_data_loaded = False
        self.is_tokenized = False
        self.ngram_range = None

    def load_data(self, text_series: pd.Series) -> None:
        self.orig_text_series = text_series
        self.is_data_loaded = True

    @staticmethod
    def _clean_text(text_series: pd.Series) -> pd.Series:
        res = text_series.str.replace(r"(<[^<>]+>|[^\w\s]+)", " ").str.lower()
        return res

    def _lemmatize(self, tokens: List[str]) -> Iterator:
        lemmatized_token_series = map(self.lemmatizer.lemmatize, tokens)
        return lemmatized_token_series

    @staticmethod
    def _remove_stopwords(lemmatized_tokens: Generator[str, None, None], stop_words: set) -> List[str]:
        res = [t for t in lemmatized_tokens if t not in stop_words]
        return res

    @staticmethod
    def _every_grams(unigrams, ngram_range: Tuple[int, int] = (1, 1)):
        res = [" ".join(x) for x in everygrams(unigrams, *ngram_range)]
        return res

    def tokenize(self, ngram_range: Tuple[int, int] = (1, 1), *, remove_stopwords: bool = True) -> None:
        if not self.is_data_loaded:
            raise Exception("text data is not loaded")
        text_series = self._clean_text(self.orig_text_series)
        token_series = text_series.apply(word_tokenize)
        unigrams = token_series.apply(self._lemmatize)
        if remove_stopwords:
            unigrams = unigrams.apply(lambda x: self._remove_stopwords(x, self.stop_words))
        self.token_series = unigrams.apply(lambda x: self._every_grams(x, ngram_range))
        self.is_tokenized = True
        self.ngram_range = ngram_range


class Prophet:
    NGRAM_RANGE = (1, 1)
    THRESHOLD = 0.145

    def __init__(self) -> None:
        self.clf = joblib.load("clf.pkl")
        self.threshold = self.THRESHOLD
        self.tokenizer = Tokenizer()
        self.tfidf = joblib.load("tfidf.pkl")

        self.tfidf.tokenizer = lambda x: x
        self.tfidf.preprocessor = lambda x: x

    @staticmethod
    def safe_predict_proba(clf: "classifier", X: np.ndarray) -> np.ndarray:
        if hasattr(clf, "predict_proba"):
            pred = clf.predict_proba(X)
        elif hasattr(clf, "decision_function"):
            pred = softmax(clf.decision_function(X))
        else:
            raise AttributeError(f"{repr(clf)} has no prediction methods")
        return pred

    @staticmethod
    def problem_predict(pred: pd.Series) -> np.ndarray:
        problem_mask = pred.apply(lambda x: x.size == 0)
        if np.any(problem_mask):
            warnings.warn(f"Empty prediction for {problem_mask.sum()} objects. Replaced with highest prob class.")
        problem_idx = np.where(problem_mask)[0]
        return problem_idx

    @staticmethod
    def get_classes(clf, row, threshold):
        return clf.classes_[row > threshold]

    def predict(self, sentence: Union[str, pd.Series]) -> pd.Series:
        if not isinstance(sentence, pd.Series):
            sentence_series = pd.Series(sentence)
        self.tokenizer.load_data(sentence_series)
        self.tokenizer.tokenize(self.NGRAM_RANGE)
        token_series = self.tokenizer.token_series
        X = self.tfidf.transform(token_series)

        pred = Prophet.safe_predict_proba(self.clf, X)

        res = pd.Series([Prophet.get_classes(self.clf, i, self.threshold) for i in pred])

        problem_idx = Prophet.problem_predict(res)
        if problem_idx.size:
            res.iloc[problem_idx] = clf.predict(X[problem_idx])
        res = ", ".join(res[0].tolist())
        return res

