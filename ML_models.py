#Written by Maja

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sys

# Libraries for preprocessing and feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Libraries for classifiers
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC

# Libraries for validation
from sklearn.model_selection import train_test_split as tts, cross_validate
from sklearn.metrics import classification_report

# Libraries with helper functions
from collections import Counter

# Libraries for visualisation
from matplotlib.colors import ListedColormap

#PATHS TO CSV FILES
hrv_lop_81k5 = r'HRV_Label_lopning_svar_81k5.csv'
hrv_lop_svar = r'HRV_Label_lopning_svar.csv'
hrv_ovrigt_svar = r'HRV_Label_ovrigt_svar.csv'
sub_lop_81k5 = r'SUBJECTIVE_Label_lopning_svar_81k5.csv' 
sub_lop_81k5_v2 = r'SUBJECTIVE_Label_lopning_svar_enskild.csv'
sub_lop_svar = r'SUBJECTIVE_Label_lopning_svar.csv' 
sub_ovrigt_svar = r'SUBJECTIVE_Label_ovrigt_svar.csv' 
sub_lop_svar_v2 = r'SUBJECTIVE_Label_lopning_svar_ver2.csv'
trimp_81k5_allt = r'TRIMP_Label_81k5_allt.csv'
trimp_lop_81k5 = r'TRIMP_Label_lopning_81k5.csv'
trimp_lop_alla = r'TRIMP_Label_lopning_alla.csv'
trimp_lop_svar_81k5 = r'TRIMP_Label_lopning_svar_81k5.csv'
trimp_lop_svar = r'TRIMP_Label_lopning_svar.csv'
trimp_ovrigt_alla = r'TRIMP_Label_ovrigt_alla.csv'
trimp_ovrigt_svar = r'TRIMP_Label_ovrigt_svar.csv'

#READ CSV AND GET VARIABLES
csv_file = sub_lop_svar_v2

df = pd.read_csv(csv_file , sep=',', engine='python')

df.rename(
    columns = {
    "Kon [1 (man)/0 (kvinna)]" : "Kön",
    "Träningsform [1 (lopning)/0 (annan)]" : "Träningsform",
    "Tid [min]" : "Tid min",
    "Medelpuls [bpm]" : "Medelpuls",
    "Maxpuls [bpm]" : "Maxpuls",
    "Trimp" : "Trimp",
    "Trimp/min [0-10]" : "Trimp/min",
    "Kalorier" : "Kalorier",
    "Distans [m]" : "Distans m",
    "Hojdstigning [m]" : "Höjdstigning m",
    "Medeltempo [min/km]" : "Medeltempo",
    "Energi [J]" : "Energi",
    "VO2 [mL/(kg*min)]" : "VO2",
    "Traningsbelastning [1-10]" : "Träningsbelastning",
    "Ovrig fysisk belastning [1-10]" : "Övrig fysisk belastning",
    "Muskeltrotthet [1-10]" : "Muskeltrötthet",
    "Mental anstrangning [1-10]" : "Mental ansträngning",
    "Skadestatus [1-10]" : "Skadestatus",
    "Sjukdomsstatus [1-10]" : "Sjukdomsstatus",
    "Somn [1-10]" : "Sömn",
    "Mat- och dryck [1-10]" : "Mat och dryck",
    "Humor [1-10]" : "Humör",
    "Upplevd aterhamtning [1-10]" : "Upplevd återhämtning",
    "Motivation [1-10]" : "Motivation",
    "HRV [RMSSD ms]" : "HRV",
    "Dagar sen mens" : "Dagar sen mens",
    "Vilopuls [bpm]" : "Vilopuls",
    "P-piller [1 (ja)/ 0 (nej)]" : "P-piller",
    "HRV-diff" : "HRV differens",
    "vilopuls-diff" : "Vilopuls differens",
    "Label [1,2,3] (h�gt v�rde == l�ngre �terh�mntning beh�vs)" : "Label"
    },
    inplace = True
)

X = df[[
    #"Kön",
    #"Träningsform",
    #"Tid min",
    #"Medelpuls",
    #"Maxpuls",
    #"Trimp",
    #"Trimp/min",
    #"Kalorier",
    #"Distans m",
    #"Höjdstigning m",
    #"Medeltempo",
    #"Energi",
    #"VO2",
    #"Träningsbelastning",
    #"Övrig fysisk belastning",
    #"Muskeltrötthet",
    #"Mental ansträngning",
    #"Skadestatus",
    #"Sjukdomsstatus",
    #"Sömn",
    #"Mat och dryck",
    #"Humör",
    #"Upplevd återhämtning",
    "Motivation",
    "HRV",
    #"Dagar sen mens",
    #"Vilopuls",
    #"P-piller",
    #"HRV differens",
    #"Vilopuls differens"
    ]]

y = df[["Label"]]
y = y.values.ravel()

#Importande select K best
find_importances = False

# IF the number of parameters is 2. What to visualise/print
printScores = False
visualise_train = False
visualise_test = False
visualise_rebalansed = True

#NORMALIZE AND SPLIT TO TRAIN AND TEST DATA

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, stratify = y)
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

#DEFINE MODELS
def chooseKNN(number):
    title = "K Nearest Neighbors"
    classifier = KNN(n_neighbors=number, metric='minkowski', p=2)
    return classifier, title


def chooseSVM():
    title = "Support Vector Machine with RBF"
    classifier = SVC(kernel='rbf', C=1)
    return classifier, title


def chooseDT():
    title = "Decision tree"
    classifier = DT(criterion="gini")
    return classifier, title


def chooseNB():
    title = "Naive Bayes with Gaussian Kernel"
    classifier = NB()
    return classifier, title

#SELECT MODEL
ChosenClassifier = chooseNB()

classifier = ChosenClassifier[0]
title = ChosenClassifier[1]

classifier.fit(X_train, y_train)

#FEATURE IMPORTANCE WITH PERMUTATION IMPORTANCE
results = permutation_importance(classifier, X_train, y_train, scoring='f1_macro')
other_importance = results.importances_mean

for i in range(1,50):
    X_train_FI, X_test_FI, y_train_FI, y_test_FI = tts(X, y, test_size = 0.2, stratify = y)
    X_train_FI = standardScaler.fit_transform(X_train_FI)
    X_test_FI = standardScaler.transform(X_test_FI)
    test_K_FI = SVC(kernel='rbf')
    test_K_FI.fit(X_train_FI, y_train_FI)
    results_in_loop = permutation_importance(classifier, X_train_FI, y_train_FI, scoring='f1_macro')
    other_importance_in_lop = results_in_loop.importances_mean
    other_importance = other_importance + other_importance_in_lop

importance = other_importance
fig_1 = plt.figure(1)
plt.title("Genomsnittlig förändring F1")
plt.barh([x for x in range(len(importance))], importance/50)
plt.yticks(ticks = np.arange(len(importance)), labels = X.columns)
plt.grid(axis='x')
for i,v in enumerate(importance):
	print(f'Feature: {X.columns[i]}, Score: {v}')


#FEATURE IMPORTANCE WITH SELECT K BEST 
if find_importances:
    N = 100
    average_values = np.zeros(np.size(X.columns))
    for i in range(N):
        X_train_ffi,X_test_ffi,y_train_ffi,y_test_ffi = tts(X.values,y,test_size=0.3,stratify=y)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier.fit(X_train_ffi, y_train_ffi)
        selector = SelectKBest(f_classif, k=10)
        selector.fit(X_train_ffi, y_train_ffi)
        scores = -np.log10(selector.pvalues_)
        average_values += scores

    for i in range(np.size(X.columns)):
        print(X.columns[i], average_values[i]/N)
    fig_2 = plt.figure(2)
    plt.title("Select K best")
    plt.barh(X.columns,scores)


#FIND AND GRAPH BEST K
k_grap = []
acc_graph = []
k_val = 0
best_k = 0

for k in range(1, 10):
    test_K = KNN(n_neighbors=k, metric='minkowski', p=2)
    test_K.fit(X_train, y_train)
    guesses = test_K.predict(X_test)
    k_accuracy = cross_validate(estimator=test_K, X=X, y=y, scoring='f1_macro', cv=3)
    print(f"{k} : {k_accuracy['test_score'].mean() * 100}")
    k_grap.append(k)
    acc_graph.append(k_accuracy['test_score'].mean() * 100)
    if k_accuracy['test_score'].mean() * 100 > k_val :
        best_k = k
        k_val = k_accuracy['test_score'].mean() * 100 

fig_3 = plt.figure(3)
plt.title("F1 värde för olika K")
plt.plot(k_grap, acc_graph)

#CONFUSION MATRIX
matrix = sklearn.metrics.plot_confusion_matrix(estimator = classifier, X = X_test, y_true = y_test)
matrix.ax_.set_title("Confusion matrix")

#PROPORTIONS
def getLargestProportions(label_y):
    a = Counter(label_y).get(2)
    c = Counter(label_y).get(3)
    a, b, c = np.int(Counter(label_y).get(1) / 2), np.int(Counter(label_y).get(2) / 2), np.int(
        Counter(label_y).get(3) * 0.8)
    return a, b, c

arg = getLargestProportions(y)

#ORIGINAL CALSSIFICATION REPORT WITHOUT OVERSAMPLING
y_hat_orig = classifier.predict(X_test)
print(y_hat_orig)
print("Original classification report")
print(classification_report(y_test, y_hat_orig))

#OVER SAMPLE MINORITY CLASS AND UNDER SAMPLE THE MAJORITY CLASSES
us = RandomUnderSampler(sampling_strategy={1: arg[0], 2: arg[1]}, random_state=13)
sm = SMOTE(sampling_strategy={3: arg[2]}, k_neighbors=1, random_state=13)

#PIPELINE TO FUNNEL SAMPLING BEFORE FITTING CLASSIFIER
print("Relationship between classes in file: ,", Counter(y))
print("Relationship in train split:", (Counter(y_train)))
print("Relationship in rebalanced data :", arg)

pip = Pipeline([('SMOTE oversampler', sm), ("Random Undersampler", us), ('classifier', classifier)], verbose=False)

#FIT AND PREDICT
pip.fit(X_train, y_train)
y_hat = pip.predict(X_test)

#PRINT CLASS REPORT
if printScores:
    print("Cross validated")
    measure = ['accuracy', 'f1_weighted', "f1_macro", "precision_micro", "recall_macro"]
    measures = np.zeros(5)
    for i in range(100):
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, stratify=y)
        X_train = standardScaler.fit_transform(X_train)
        X_test = standardScaler.transform(X_test)
        classifier = SVC()
        classifier.fit(X_train,y_train)
        pip = Pipeline([('SMOTE oversampler', sm), ("Random Undersampler", us), ('classifier', classifier)],
                       verbose=False)
        pip.fit(X_train,y_train)
        accuracies = cross_validate(estimator=classifier, X=X_test,
                                    y=y_test, scoring=measure, cv=2)
        y_hat = classifier.predict(X_test)
        print(classification_report(y_test,y_hat))
        measures[0] += accuracies['test_precision_micro'].mean() * 100
        measures[1] += accuracies['test_recall_macro'].mean() * 100
        measures[2] += accuracies['test_accuracy'].mean() * 100
        measures[3] += accuracies['test_f1_macro'].mean() * 100
        measures[4] += accuracies['test_f1_weighted'].mean() * 100
        print(measures/(i+1))
        if i > 98:
            print("Accuracy: {:.3f} %".format(accuracies['test_accuracy'].mean() * 100),
            ', Std: {:.3f} %'.format(accuracies['test_accuracy'].std() * 100))
            print("Weighted F1: {:.3f} %".format(accuracies['test_f1_weighted'].mean() * 100),
            ", Std: {:.3f} %".format(accuracies['test_f1_weighted'].std() * 100))
            print("Macro F1: {:.3f} %".format(accuracies['test_f1_macro'].mean() * 100),
            ", Std: {:.3f} %".format(accuracies['test_f1_macro'].std() * 100))
            print("Macro Precision: {:.3f} %".format(accuracies['test_precision_micro'].mean() * 100),
            ", Std: {:.3f} %".format(accuracies['test_precision_micro'].std() * 100))
            print("Macro Recall: {:.3f} %".format(accuracies['test_recall_macro'].mean() * 100),
            ", Std: {:.3f} %".format(accuracies['test_recall_macro'].std() * 100))
    print(measures/100)

#VISUALISE
if visualise_train:
    if np.size(np.unique(X.columns)) != 2:
        print("\n No visualisation available: Features must be 2")
        sys.exit(0)

    X_set, y_set = X_train, y_train
    print(Counter(y_set))
    min_xaxis = X_set[:, 0].min() - 1
    max_xaxis = X_set[:, 0].max() + 1
    min_yaxis = X_set[:, 1].min() - 1
    max_yaxis = X_set[:, 1].max() + 1

    X1, X2 = np.meshgrid(np.arange(min_xaxis, max_xaxis, step=0.01),
                         np.arange(min_yaxis, max_yaxis, step=0.01))

    predict = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    fig_5 = plt.figure(5)
    plt.contourf(X1, X2, predict, alpha=0.9, cmap=ListedColormap(('green', 'orange', 'red')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    labels = ["Kortare", "Medellång", "Längre"]
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('green', 'orange', 'red'))(i),
                    edgecolors="black",
                    label=labels[i])
    title1 = title + " [training set]"
    plt.title(title1)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.legend(title="Återhämtning")

if visualise_test:
        if np.size(np.unique(X.columns)) != 2:
            print("\n No visualisation available: Features must be 2")
            sys.exit(0)

        X_set, y_set = X_train, y_train
        print(Counter(y_set))
        min_xaxis = X_set[:, 0].min() - 1
        max_xaxis = X_set[:, 0].max() + 1
        min_yaxis = X_set[:, 1].min() - 1
        max_yaxis = X_set[:, 1].max() + 1

        X1, X2 = np.meshgrid(np.arange(min_xaxis, max_xaxis, step=0.01),
                             np.arange(min_yaxis, max_yaxis, step=0.01))

        predict = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
        fig_6 = plt.figure(6)
        plt.contourf(X1, X2, predict, alpha=0.9, cmap=ListedColormap(('green', 'orange', 'red')))

        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        labels = ["Kortare", "Medellång", "Längre"]
        for i, j in enumerate(np.unique(y_test)):
            plt.scatter(X_test[y_test == j, 0], X_test[y_test == j, 1],
                        c=ListedColormap(('green', 'orange', 'red'))(i),
                        edgecolors="black",
                        label=labels[i])
        title = title + " [test set]"
        plt.title(title)
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.legend(title="Återhämtning")

#VISUALISE REBALANCED
if visualise_rebalansed:

    if np.size(np.unique(X.columns)) != 2:
        print("No visualisation available: Features must be 2")
        sys.exit(0)
    X_res, y_res = us.fit_resample(X_train, y_train)
    X_set, y_set = sm.fit_resample(X_res, y_res)

    min_xaxis = X_set[:, 0].min() - 1
    max_xaxis = X_set[:, 0].max() + 1
    min_yaxis = X_set[:, 1].min() - 1
    max_yaxis = X_set[:, 1].max() + 1

    X1, X2 = np.meshgrid(np.arange(min_xaxis, max_xaxis, step=0.01),
                         np.arange(min_yaxis, max_yaxis, step=0.01))

    predict = pip.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    fig_7 = plt.figure(7)
    plt.contourf(X1, X2, predict, alpha=0.9, cmap=ListedColormap(('green', 'orange', 'red')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    labels = ["Kortare", "Medellång", "Längre"]
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('green', 'orange', 'red'))(i),
                    edgecolors="black",
                    label=labels[i])
    plt.title(title)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.legend(title="Återhämtning")

#PLOT ALL GRAPHS
plt.show()
