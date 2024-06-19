import numpy as np


from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import PorterStemmer


class Pretraitement:

    def __init__(self):
        pass

    def data_stemming(self,data):
        ps = PorterStemmer()
        result = []
        for i in range(len(data)):
            words = data[i]

            stemmed_words = [ps.stem(word) for word in words]
           # print(stemmed_words)
            result.append(" ".join(stemmed_words))
        return result

    def tokenizing(self,data):
        result = []
        for i in range(len(data)):
            words = data[i].split()
            cleaned_words = []
            for word in words:
                for symbol in word:
                    if not symbol.isalpha():
                        word = word.replace(symbol, "")
                cleaned_words.append(word)
            result.append(cleaned_words)

        return result



    def separeData2TrainAndTest(self,dataOrLabels):
        separationPoint = int(len(dataOrLabels) * 0.7)
        training_set = dataOrLabels[:separationPoint]
        testing_set = dataOrLabels[separationPoint:]
        return (training_set, testing_set)

    def vectorizing(self, data, vectorizer, fit = True):
        if fit: X = vectorizer.fit_transform(data)
        else: X = vectorizer.transform(data)

        return X

