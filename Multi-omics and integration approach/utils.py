import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
import numpy as np
import itertools
warnings.simplefilter('ignore')

def create_dataframe():
    """
    The function performs a dataset creation and some preprocessing tasks. In particular it:
    - loads the 3 omnics dataset;
    - sets them in the correct format (transpose the matrices);
    - nomalizes them;
    - load the corresponding labels;

    :return: the three omnics dataset + the corresponding labels
    """
    # the features are on the row axis while the samples are on the columns
    transcriptome_df = pd.read_csv('./simulated_dataset/mRNA.txt', sep='\t', header=0).transpose()
    genome_df = pd.read_csv('./simulated_dataset/meth.txt', sep='\t', header=0).transpose()
    proteome_df = pd.read_csv('./simulated_dataset/prot.txt', sep='\t', header=0).transpose()

    # set the name of the columns
    transcriptome_df.columns = transcriptome_df.loc['probe'].values
    transcriptome_df.drop('probe', inplace=True, axis=0)
    genome_df.columns = genome_df.loc['probe'].values
    genome_df.drop('probe', inplace=True, axis=0)
    proteome_df.columns = proteome_df.loc['probe'].values
    proteome_df.drop('probe', inplace=True, axis=0)

    # normalizing the dataset
    mean, std = transcriptome_df.mean(axis=0), transcriptome_df.std(axis=0)
    transcriptome_df = (transcriptome_df - mean) / (std + 10 ** (-6))

    mean, std = genome_df.mean(axis=0), genome_df.std(axis=0)
    genome_df = (genome_df - mean) / (std + 10 ** (-6))

    mean, std = proteome_df.mean(axis=0), proteome_df.std(axis=0)
    proteome_df = (proteome_df - mean) / (std + 10 ** (-6))

    print(f'Length of the transcriptome dataframe: {transcriptome_df.shape}')
    print(f'Length of the genome dataframe: {genome_df.shape}')
    print(f'Length of the proteome dataframe: {proteome_df.shape}')

    labels_df = pd.read_csv('./simulated_dataset/clusters.txt', sep='\t', header=0).set_index('subjects')

    return transcriptome_df, genome_df, proteome_df, labels_df


def hyperparameter_tuning(X_train, y_train, X_test, y_test, verbose=True):
    """
    The function performs an hyperparameter tuning for 4 different classifiers (MLPClassifier,
    KNeighborsClassifier, RandomForestClassifier, LogisticRegression).

    :param X_train: the training dataset in the format [n_samples, n_fts]
    :param y_train: the training labels in the format [n_samples]
    :param X_test: the test dataset in the format [n_samples, n_fts]
    :param y_test: the test labels in the format [n_samples]
    :param verbose: flag that control the possibility to print or not intermediate results
    :return: the best classifier among the three
    """
    # implementing 4 different classifiers
    clfs = [MLPClassifier(), KNeighborsClassifier(), RandomForestClassifier(), LogisticRegression()]
    # parameters
    params = [
        {
            'hidden_layer_sizes': [(50,), (100,)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'solver': ['adam']
        }, {
            'n_neighbors': [1, 5, 20, 50, 100]
        }, {
            'n_estimators': [20, 100, 200],
            'criterion': ['gini', 'entropy'],
            'oob_score': [True],

        }, {
            'C': [0.001, 0.01, 0.1, 1],
        }
    ]

    best_clf = None
    best_score = 0
    for clf, param in zip(clfs, params):


        grid_search = GridSearchCV(clf, param, cv=5)
        grid_search.fit(X_train, y_train)
        if verbose:
            print(f'{type(clf).__name__}')
            print(f'Best configuration: {grid_search.best_params_}')
            print(f'Accuracy on validation set (5 folds): {grid_search.best_score_}')
            print()

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_clf = grid_search.best_estimator_



    print(f'Best classifier: {type(best_clf).__name__}')
    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on the test set: {accuracy}')

    return best_clf

def clustering_accuracy(y_true, y_pred):
    """
     The function seek to determines the accuracy score even in the case in
     which the clustering assign a wrong name of class by merge togheter the right data points.

    :param y_true: the ground truth
    :param y_pred: the preducted labels
    :return: the accuracy score
    """
    classes = np.unique(y_true)

    permutations = itertools.permutations(classes, len(classes))
    best_accuracy = 0
    best_combination = None
    for p in permutations:

        new_combination = dict(zip(classes, p))
        n_correct = 0

        for (y_true_class, y_pred_class) in new_combination.items():
            n_correct += len(np.where((y_true == y_true_class)*(y_pred == y_pred_class))[0])

        accuracy = n_correct / len(y_true)

        if accuracy > best_accuracy:
            best_combination = new_combination
            best_accuracy = accuracy

    return best_accuracy, best_combination


def soft_clustering_weights(X, centroids, best_combination):
    """
    Function to calculate the weights from soft k-means
    :param X : Array of data. shape = N x F, for N data points and F Features
    :param centroids : Array of cluster centres. shape = Nc x F, for Nc number of clusters. Input kmeans.cluster_centres_ directly.
    """
    n_clusters = centroids.shape[0]

    #  computes the overall distance for each samples
    # Get distances from the cluster centres for each data point and each cluster
    all_distances = []


    for c in sorted(best_combination):
        # this way I consider all the classes in the right order of the
        # ground truth
        j = best_combination[c]-1
        distances = (np.sum((X - centroids[j, :])**2, axis=1))**(1/2)
        all_distances.append(distances)

    all_distances = np.array(all_distances)
    sum_distances = np.sum(all_distances, axis=0) #(n_samples, classes)
    ratio_distances = []

    for j in range(n_clusters):
        ratio_distances.append(all_distances[j]/sum_distances)

    ratio_distances = np.array(ratio_distances).T
    weights = 1./ratio_distances
    weights = (weights.T / np.sum(weights, axis=1)).T

    # (500, 5)
    return weights



