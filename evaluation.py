from sklearn.model_selection import cross_val_score
from  sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import  LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV


'''models from sklearn, grid search or cv'''

def parameter_eval_adaboost(base_estimator,train_features,labels,grid_params,cv):
    print("Grid Search")
    best_clf = GridSearchCV(AdaBoostClassifier(base_estimator=base_estimator), grid_params, cv=cv)
    print("fitting")
    best_clf.fit(train_features, labels)
    print("Best parameters set found on development set:")
    print()
    print(best_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = best_clf.cv_results_['mean_test_score']
    stds = best_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, best_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(best_clf.get_params())
    return best_clf.best_estimator_


def parameter_eval_log_reg(train_features,labels,grid_params,cv):
    print("Grid Search")
    best_clf = GridSearchCV(LogisticRegression(), grid_params, cv=cv)
    print("fitting")
    best_clf.fit(train_features, labels)
    print("Best parameters set found on development set:")
    print()
    print(best_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = best_clf.cv_results_['mean_test_score']
    stds = best_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, best_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(best_clf.get_params())
    return best_clf.best_estimator_


def parameter_eval_tree(train_features,labels,grid_params,cv):
    best_clf = GridSearchCV(tree.DecisionTreeClassifier(), grid_params, cv=cv)
    print("fitting")
    best_clf.fit(train_features, labels)
    print("Best parameters set found on development set:")
    print()
    print(best_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = best_clf.cv_results_['mean_test_score']
    stds = best_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, best_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(best_clf.get_params())
    return best_clf.best_estimator_

def parameter_eval_random_forest(train_features,labels,grid_params,cv):
    print("Grid Search")
    best_clf = GridSearchCV(RandomForestClassifier(), grid_params, cv=cv)
    print("fitting")
    best_clf.fit(train_features, labels)
    print("Best parameters set found on development set:")
    print()
    print(best_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = best_clf.cv_results_['mean_test_score']
    stds = best_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, best_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(best_clf.get_params())
    return best_clf.best_estimator_



def parameter_eval_random_forest(train_features,labels,grid_params,cv):
    print("Grid Search")
    best_clf = GridSearchCV(RandomForestClassifier(), grid_params,cv = cv )
    best_clf.fit(train_features,labels)
    print("Best parameters set found on development set:")
    print()
    print(best_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = best_clf.cv_results_['mean_test_score']
    stds = best_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, best_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(best_clf.get_params())
    return best_clf.best_estimator_

def evaluate_log_loss(clf,train_features,labels,cv):
    scores = cross_val_score(clf, train_features, labels,scoring="neg_log_loss",cv=cv)
    print ("CV Scores")
    print (scores)
    cv_score = np.mean(scores)
    return cv_score


def evaluate_accuracy(clf, train_features, labels,cv):
    scores = cross_val_score(clf, train_features, labels,scoring="accuracy",cv=cv)
    print ("CV Scores")
    print (scores)
    cv_score = np.mean(scores)
    return cv_score
