from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import tree, svm
import time
import nltk
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier

import Pretraitement

import DataReader

class Tache:
    def __init__(self):
        self.start_time = None


    def read_file(self, file_path = 'offenseval-training-v1.tsv'):
        reader = DataReader.DataReader(file_path)
        data, labels = reader.get_labelled_data()
        return (data, labels)

    def multinomial_bayes(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = MultinomialNB()
        clf.fit(vect_train, train_label)

        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def gaussian_bayes(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        gnb = GaussianNB() #bayes gaussian
        gnb.fit(vect_train.toarray(), train_label)

        print("temps pour entrainement: ")
        self.timeCalcul()
        score = gnb.score(vect_test.toarray(), test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def decision_tree(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = tree.DecisionTreeClassifier(criterion="gini")
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def decision_tree_h20(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = tree.DecisionTreeClassifier(max_depth=50)
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def decision_tree_entropy(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = tree.DecisionTreeClassifier(criterion="entropy") #gini < entropy
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def random_forest(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = RandomForestClassifier(n_estimators = 100)
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def random_forest_h100(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = RandomForestClassifier(max_depth=100)
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def random_forest_n15(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = RandomForestClassifier(n_estimators = 15)
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score


    def svm_linear(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = svm.SVC(kernel='linear')
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def svm_rbf(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = svm.SVC(kernel='rbf')
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def mlp_logistic(self,vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = MLPClassifier(activation='logistic',hidden_layer_sizes=(100,))
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def mlp_relu(self, vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = MLPClassifier(activation='relu', hidden_layer_sizes= (100,))
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def mlp_relu_30layers(self, vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = MLPClassifier(activation='relu', hidden_layer_sizes= (30,))
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score
    def mlp_relu_multiple_couches(self, vect_train, vect_test, train_label, test_label):
        self.timeCalcul()
        clf = MLPClassifier(activation='relu', hidden_layer_sizes= (100,20,10))
        clf.fit(vect_train, train_label)
        print("temps pour entrainement: ")
        self.timeCalcul()
        score = clf.score(vect_test, test_label)
        print("temps pour evaluer: ")
        self.timeCalcul()
        return score

    def timeCalcul(self): #fonction pour evaluer le temps d'execution
        if self.start_time is None:
            self.start_time = time.time()
        else:
            end_time = time.time()
            temps = end_time - self.start_time
            print("temps pour ce algorithme est ", temps, " secondes \n")
            self.start_time = time.time()



    def tache_base(self):
        nltk.download("punkt")

        pretrait = Pretraitement.Pretraitement()
        data, labels = tache.read_file()

        tokenized_data = pretrait.tokenizing(data)
        data = pretrait.data_stemming(tokenized_data)

        train_data, test_data = pretrait.separeData2TrainAndTest(data)
        train_labels, test_labels = pretrait.separeData2TrainAndTest(labels)

        vectorizer = CountVectorizer(stop_words='english')
        #vectorizer = TfidfVectorizer(stop_words= 'english')

        X_train = pretrait.vectorizing(train_data, vectorizer)
        X_test = pretrait.vectorizing(test_data, vectorizer, fit=False)
        print("tache de base: \n\n")

        print('Analyse de temps de calculs: \n')

        print('native bayes:')

        NaiveBayes = self.multinomial_bayes(X_train,X_test, train_labels, test_labels)

        print('arbre de decision:')
        self.start_time = None
        arbreDecision = self.decision_tree(X_train,X_test, train_labels, test_labels)

        print('foret aleatoire:')
        self.start_time = None
        foretAleatoire = self.random_forest(X_train,X_test, train_labels, test_labels)

        self.start_time = None
        print('svm:')
        svmModel = self.svm_rbf(X_train,X_test, train_labels, test_labels)

        print('mlp:')
        self.start_time = None
        multipleLayer = self.mlp_relu(X_train,X_test, train_labels, test_labels)



        print("les precisions des algorithmes sont: \n\n")


        print("Naive Bayes algorithme: \n")
        print(NaiveBayes,'\n\n')
        print("arbre decision algorithme: \n")
        print(arbreDecision, '\n\n')
        print("foret aleatoire algorithme: \n")
        print(foretAleatoire, '\n\n')
        print("SVM algorithme: \n")
        print(svmModel, '\n\n')
        print("multipleLayer algorithme: \n")
        print(multipleLayer)




    def tache_exploration(self):
        nltk.download("punkt")
        pretrait = Pretraitement.Pretraitement()
        data, labels = tache.read_file()

        tokenized_data = pretrait.tokenizing(data)
        data = pretrait.data_stemming(tokenized_data)

        train_data, test_data = pretrait.separeData2TrainAndTest(data)
        train_labels, test_labels = pretrait.separeData2TrainAndTest(labels)

        vectorizer = CountVectorizer(stop_words='english')
        #vectorizer = TfidfVectorizer(stop_words= 'english')

        X_train = pretrait.vectorizing(train_data, vectorizer)
        X_test = pretrait.vectorizing(test_data, vectorizer, fit=False)
        print("tache d'exploration: \n\n")

        print('Analyse de temps de calculs: \n')

        print('Multinomial native bayes:')

        naiveBayes = self.multinomial_bayes(X_train,X_test, train_labels, test_labels)

        print('Gaussien native bayes:')

        self.start_time = None
        gaussien = self.multinomial_bayes(X_train, X_test, train_labels, test_labels)

        print('arbre de decision defaut:')

        self.start_time = None
        arbreDecision = self.decision_tree(X_train,X_test, train_labels, test_labels)

        print('arbre de decision entropy:')

        self.start_time = None
        arbreDecision_entropy = self.decision_tree_entropy(X_train, X_test, train_labels, test_labels)

        print('arbre de decision h = 20:')

        self.start_time = None
        arbreDecision_h20 = self.decision_tree_h20(X_train, X_test, train_labels, test_labels)

        print('foret aleatoire defaut:')
        self.start_time = None
        foretAleatoire = self.random_forest(X_train,X_test, train_labels, test_labels)


        print('foret aleatoire n = 15:')
        self.start_time = None
        foretAleatoire_n15 = self.random_forest_n15(X_train,X_test, train_labels, test_labels)

        print('foret aleatoire h = 100:')
        self.start_time = None
        foretAleatoire_h100 = self.random_forest_h100(X_train, X_test, train_labels, test_labels)

        self.start_time = None
        print('svm lineaire:')
        svmModel = self.svm_linear(X_train,X_test, train_labels, test_labels)

        self.start_time = None
        print('svm rbf:')
        svmModel_rbf = self.svm_rbf(X_train, X_test, train_labels, test_labels)

        print('mlp relu 30 layers:')
        self.start_time = None
        multipleLayer_30 = self.mlp_relu_30layers(X_train, X_test, train_labels, test_labels)
        print(multipleLayer_30)


        print('mlp relu:')
        self.start_time = None
        multipleLayer = self.mlp_relu(X_train,X_test, train_labels, test_labels)

        print('mlp logistic:')
        self.start_time = None
        multipleLayer_logistic = self.mlp_logistic(X_train, X_test, train_labels, test_labels)

        print('mlp relu avec multiples couches:')
        self.start_time = None
        multipleLayer_multpile = self.mlp_relu_multiple_couches(X_train, X_test, train_labels, test_labels)

        print("les precisions des algorithmes sont: \n\n")

        print("Multinomial Naive Bayes algorithme: \n")
        print(naiveBayes,'\n\n')

        print("Gaussien Naive Bayes algorithme: \n")
        print(gaussien, '\n\n')

        print("arbre decision algorithme: \n")
        print(arbreDecision, '\n\n')

        print("arbre decision entropy algorithme: \n")
        print(arbreDecision_entropy, '\n\n')


        print("arbre decision max_h = 20 algorithme: \n")
        print(arbreDecision_h20, '\n\n')

        print("foret aleatoire algorithme: \n")
        print(foretAleatoire, '\n\n')


        print("foret aleatoire n = 15 algorithme: \n")
        print(foretAleatoire_n15, '\n\n')


        print("foret aleatoire max_h = 100 algorithme: \n")
        print(foretAleatoire_h100, '\n\n')

        print("SVM lineaire algorithme: \n")
        print(svmModel, '\n\n')

        print("SVM rbf algorithme: \n")
        print(svmModel_rbf, '\n\n')



        print("multipleLayer relu 30 layers algorithme: \n")
        print(multipleLayer_30)

        print("multipleLayer relu algorithme: \n")
        print(multipleLayer)


        print("multipleLayer logistic algorithme: \n")
        print(multipleLayer_logistic)

        print("multipleLayer avec plusieurs couches algorithme: \n")
        print(multipleLayer_multpile)


if __name__ == "__main__":
    tache = Tache()
    tache.tache_exploration()
    #tache.tache_base()



# 0 - Not offensive
# 1 - Offensive
