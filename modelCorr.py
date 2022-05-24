import configparser
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, \
    classification_report, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
dataset_filepath = config['dataset_clean_file']
correlation_file = config['correlation_target_file']


# dropped columns with variance lower than 0.1
def prepare_csv_file():
    filter_dataset_file = pd.read_csv(filepath_or_buffer=dataset_filepath, delimiter=',')

    calculate_correlation = filter_dataset_file.corr()
    columns_name = []
    for row_keys, row_values in calculate_correlation.iterrows():
        if abs(row_values.iloc[-1]) < 0.1:
            columns_name.append(row_keys)

    filter_dataset_file.drop(columns_name, axis=1, inplace=True)
    filter_dataset_file.to_csv(correlation_file, index=False, sep=',')
    print('Number of inappropriate columns: ' + str(len(columns_name)))
    return


# model based on MLP
def model_mlp():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=(150, 100, 50), activation='tanh', solver='adam', alpha=0.05,
                       learning_rate_init=0.001, learning_rate='adaptive', max_iter=5000)

    mlp.fit(X_train, y_train)
    predicted_target = mlp.predict(X_test)

    mse = mean_squared_error(y_test, predicted_target)
    rmse = mean_squared_error(y_test, predicted_target, squared=False)
    mae = mean_absolute_error(y_test, predicted_target)
    r2 = r2_score(y_test, predicted_target)

    print('Mean Squared Error: ' + str(mse))
    print('Root Mean Squared Error: ' + str(rmse))
    print('Mean Absolute Error: ' + str(mae))
    print('R2 score: ' + str(r2))
    return


# model based on SVR
def model_svr():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=42)
    svr = SVR(kernel='linear', gamma='scale', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    predicted_target = svr.predict(X_test)

    mse = mean_squared_error(y_test, predicted_target)
    rmse = mean_squared_error(y_test, predicted_target, squared=False)
    mae = mean_absolute_error(y_test, predicted_target)
    r2 = r2_score(y_test, predicted_target)

    print('Mean Squared Error: ' + str(mse))
    print('Root Mean Squared Error: ' + str(rmse))
    print('Mean Absolute Error: ' + str(mae))
    print('R2 score: ' + str(r2))
    return


# model based on Decision Trees
def model_decision_tree():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=453)

    parameters = {
        'estimator__max_features': [None, 'auto', 'log2', 'sqrt'],
        'estimator__max_depth': [5, 10, 15, 20, 50, 70, 90, 120, 150, 200],
        'estimator__min_samples_split': [5, 10, 15],
        'estimator__min_samples_leaf': [2, 3, 4, 5, 6]
    }

    model = OneVsRestClassifier(DecisionTreeClassifier())
    model_grid_search = GridSearchCV(estimator=model, param_grid=parameters, verbose=3)
    model_grid_search.fit(X_train, y_train)
    prediction_results = model_grid_search.predict(X_test)
    probalility_results = model_grid_search.predict_proba(X_test)

    confusion_matrix_values = confusion_matrix(y_test, prediction_results, labels=categories)
    classification_report_all = classification_report(y_test, prediction_results, zero_division=1)

    accuracy_result = accuracy_score(y_test, prediction_results)
    f1_result = f1_score(y_test, prediction_results, average='macro')
    precision_result = precision_score(y_test, prediction_results, average='macro')
    recall_result = recall_score(y_test, prediction_results, average='macro')

    print('Accuracy: ' + str(accuracy_result))
    print('F1 score is:' + str(f1_result))
    print('Precision score: ' + str(precision_result))
    print('Recall score: ' + str(recall_result))
    print('Confusion matrix')
    print(confusion_matrix_values)
    print(classification_report_all)

    roc_auc_value(probalility_results, y_test)
    confusion_matrix_plot(confusion_matrix_values, 'decision-tree-corr-dataset')
    return


# model based on C-Support Vector
def model_SVC():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=453)

    parameters = {
        'estimator__C': [0.1, 1, 10, 100, 1000],
        'estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    model = OneVsRestClassifier(SVC(probability=True))
    model_grid_search = GridSearchCV(estimator=model, param_grid=parameters, verbose=3)
    model_grid_search.fit(X_train, y_train)
    prediction_results = model_grid_search.predict(X_test)
    probalility_results = model_grid_search.predict_proba(X_test)

    confusion_matrix_values = confusion_matrix(y_test, prediction_results, labels=categories)
    classification_report_all = classification_report(y_test, prediction_results, zero_division=1)

    accuracy_result = accuracy_score(y_test, prediction_results)
    f1_result = f1_score(y_test, prediction_results, average='macro')
    precision_result = precision_score(y_test, prediction_results, average='macro')
    recall_result = recall_score(y_test, prediction_results, average='macro')

    print('Accuracy: ' + str(accuracy_result))
    print('F1 score is:' + str(f1_result))
    print('Precision score: ' + str(precision_result))
    print('Recall score: ' + str(recall_result))
    print('Confusion matrix')
    print(confusion_matrix_values)
    print(classification_report_all)

    roc_auc_value(probalility_results, y_test)
    confusion_matrix_plot(confusion_matrix_values, 'svc-corr-dataset')
    return


# model based on K-Nearest Neighbour
def model_knn():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=453)

    parameters = {
        'estimator__n_neighbors': [1, 2, 3, 4, 5, 7, 10],
        'estimator__weights': ['uniform', 'distance'],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    }

    model = OneVsRestClassifier(KNeighborsClassifier())
    model_grid_search = GridSearchCV(estimator=model, param_grid=parameters, verbose=3)
    model_grid_search.fit(X_train, y_train)
    prediction_results = model_grid_search.predict(X_test)
    probalility_results = model_grid_search.predict_proba(X_test)

    confusion_matrix_values = confusion_matrix(y_test, prediction_results, labels=categories)
    classification_report_all = classification_report(y_test, prediction_results, zero_division=1)

    accuracy_result = accuracy_score(y_test, prediction_results)
    f1_result = f1_score(y_test, prediction_results, average='macro')
    precision_result = precision_score(y_test, prediction_results, average='macro')
    recall_result = recall_score(y_test, prediction_results, average='macro')

    print('Accuracy: ' + str(accuracy_result))
    print('F1 score is:' + str(f1_result))
    print('Precision score: ' + str(precision_result))
    print('Recall score: ' + str(recall_result))
    print('Confusion matrix')
    print(confusion_matrix_values)
    print(classification_report_all)

    roc_auc_value(probalility_results, y_test)
    confusion_matrix_plot(confusion_matrix_values, 'knn-corr-dataset')
    return


# model based on Random Forest
def model_random_forest():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=453)

    parameters = {
        'estimator__n_estimators': [100, 150, 200],
        'estimator__max_features': ['sqrt', 'log2'],
        'estimator__max_depth': [2, 6, 8, 10],
    }

    model = OneVsRestClassifier(RandomForestClassifier())
    model_grid_search = GridSearchCV(estimator=model, param_grid=parameters, verbose=3)
    model_grid_search.fit(X_train, y_train)
    prediction_results = model_grid_search.predict(X_test)
    probalility_results = model_grid_search.predict_proba(X_test)

    confusion_matrix_values = confusion_matrix(y_test, prediction_results, labels=categories)
    classification_report_all = classification_report(y_test, prediction_results, zero_division=1)

    accuracy_result = accuracy_score(y_test, prediction_results)
    f1_result = f1_score(y_test, prediction_results, average='macro')
    precision_result = precision_score(y_test, prediction_results, average='macro')
    recall_result = recall_score(y_test, prediction_results, average='macro')

    print('Accuracy: %f' + str(accuracy_result))
    print('F1 score is:' + str(f1_result))
    print('Precision score: ' + str(precision_result))
    print('Recall score: ' + str(recall_result))
    print('Confusion matrix')
    print(confusion_matrix_values)
    print(classification_report_all)

    roc_auc_value(probalility_results, y_test)
    confusion_matrix_plot(confusion_matrix_values, 'random_forest-corr-dataset')
    return


# model based on Gradient Boost
def model_gradient_boosting():
    dataset_file = pd.read_csv(filepath_or_buffer=correlation_file, delimiter=',')
    all_feature = dataset_file.drop(labels=['result'], axis=1)
    target_class = dataset_file['result']
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    X_train, X_test, y_train, y_test = train_test_split(all_feature, target_class, test_size=0.25, shuffle=True,
                                                        random_state=453)

    parameters = {
        'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'estimator__n_estimators': [100, 200, 300, 400],
        'estimator__max_features': ['auto', 'sqrt', 'log2'],
    }

    model = OneVsRestClassifier(GradientBoostingClassifier())
    model_grid_search = GridSearchCV(estimator=model, param_grid=parameters, verbose=3)
    model_grid_search.fit(X_train, y_train)
    prediction_results = model_grid_search.predict(X_test)
    probalility_results = model_grid_search.predict_proba(X_test)

    confusion_matrix_values = confusion_matrix(y_test, prediction_results, labels=categories)
    classification_report_all = classification_report(y_test, prediction_results, zero_division=1)

    accuracy_result = accuracy_score(y_test, prediction_results)
    f1_result = f1_score(y_test, prediction_results, average='macro')
    precision_result = precision_score(y_test, prediction_results, average='macro')
    recall_result = recall_score(y_test, prediction_results, average='macro')

    print('Accuracy: ' + str(accuracy_result))
    print('F1 score is:' + str(f1_result))
    print('Precision score: ' + str(precision_result))
    print('Recall score: ' + str(recall_result))
    print('Confusion matrix')
    print(confusion_matrix_values)
    print(classification_report_all)

    roc_auc_value(probalility_results, y_test)
    confusion_matrix_plot(confusion_matrix_values, 'gradient-boosting-corr-dataset')
    return


# create ROC-AUC graph
def roc_auc_value(probability_target, target_results):
    probability_target = np.where(probability_target > np.inf, np.nan, probability_target)
    roc_auc_calculate = roc_auc_score(target_results, probability_target, multi_class='ovr')
    print('ROC AUC score: ' + str(roc_auc_calculate))
    return


# create confusion matrix
def confusion_matrix_plot(confusion_matrix_values, name_picture):
    labels = ["".join("c" + str(i)) for i in range(1, 17)]
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(confusion_matrix_values, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('Confusion_heatmap-{}.png'.format(name_picture))
    return
