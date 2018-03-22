import numpy as np
from data_loader import load_linearized, load_data_2d
from feature_extraction import calc_PCA, get_PCA_features, get_dct_features_2dinput,get_dct_features_ulc
from feature_extraction import transform_linear_scale
import evaluation
from sklearn import tree, svm, linear_model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from submission_writer import write_submission, write_params
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

def do_experiment(feature_choice, percentage,dct_n, grid_search, grid_params):
    if feature_choice == "PCA":
        print(feature_choice)
        band1, band2, angles, labels = load_linearized("train.json", True)
        pca_1 = calc_PCA(band1,percentage[0])
        pca_2 = calc_PCA(band2, percentage[1])
        #band1 = transform_linear_scale(band1)
        #band2 = transform_linear_scale(band2)
        pca1_train = get_PCA_features(pca_1, band1)
        pca2_train = get_PCA_features(pca_2, band2)
        train_feats = np.hstack((pca1_train, pca2_train))
        print(train_feats.shape)
        print("test")
        band1, band2, _, ids = load_linearized("test.json", False)
        print("loaded")
        #band1 = transform_linear_scale(band1)
        #band2 = transform_linear_scale(band2)
        pca1_test = get_PCA_features(pca_1, band1)
        pca2_test = get_PCA_features(pca_2, band2)
        test_feats = np.hstack((pca1_test, pca2_test))

    if feature_choice == "DCT":
        print(feature_choice)
        band1, band2, angles, labels = load_data_2d("train.json", True)
        ##band1 = transform_linear_scale(band1)
        ##band2 = transform_linear_scale(band2)
        dct_train1 = get_dct_features_ulc(band1, dct_n)
        dct_train2 = get_dct_features_ulc(band2, dct_n)
        train_feats = np.hstack((dct_train1, dct_train2))
        print(train_feats.shape)
        print("test")
        band1, band2, _, ids = load_data_2d("test.json", False)
        ##band1 = transform_linear_scale(band1)
        ##band2 = transform_linear_scale(band2)

        print("loaded")
        dct_test1 = get_dct_features_ulc(band1, dct_n)
        dct_test2 = get_dct_features_ulc(band2, dct_n)
        test_feats = np.hstack((dct_test1, dct_test2))

    if grid_search :
        print ("Grid Search")
        clf = evaluation.parameter_eval_random_forest(train_feats, labels, grid_params, cv=3)
        #joblib.dump(clf,'best_clf.pkl')
    else:
        print("NO Grid Search")
        #clf = tree.DecisionTreeClassifier()
        clf = RandomForestClassifier (n_estimators=600,criterion="gini")
        #clf = joblib.load("best_clf.pkl")
        #clf = linear_model.LogisticRegression()
    print("Cross_eval")
    cv_score = evaluation.evaluate_accuracy(clf,train_feats, labels,3)
    print("fitting")
    clf.fit(train_feats,labels)
    print("evaluate")
    preds = clf.predict_proba(test_feats)
    prob0 = preds[:,0]
    prob1  = preds[:,1]
    preds = [prob1[i] if 1 - prob1[i] < 1 -prob0[i] else 1-prob0[i] for i in range(prob0.shape[0])]
    preds = np.array(preds, dtype=np.float16)
    print ("smallest")
    print ((np.sort(preds))[:10])
    print("biggest")
    print ((np.sort(preds)[::-1])[:10])

    # predict proba gives probability y is 0 or 1 -> wrong -> solution np.where
    ## preds = np.amax(preds,axis=1)

    write_submission(ids,preds)
    write_params(clf,name="Log Reg",features="DCT 20 ULC", cv_score=cv_score)


def main():
    #grid_params = [{'penalty': ["l2"], 'dual': [True, False], "C": [0.01, 0.1, 1, 10, 20, 100]},
    #               {'penalty': ["l1"], 'dual': [False], "C": [0.01, 0.1, 1, 10, 20, 100]}, ]

    grid_params = [   {'n_estimators': [100,200,300,400,500,600,700], 'criterion': ['gini'],
                       'max_features': ['auto','sqrt', 'log2']},
       {'n_estimators': [100,200,300,400,500,600,700] , 'criterion': ['entropy'],
                      'max_features': ['auto', 'sqrt', 'log2']},]


    do_experiment(feature_choice="DCT", percentage=(0.75,0.55), dct_n= 20,grid_search=True, grid_params= grid_params)

if __name__ == "__main__":
    main()
        ##grid_params = [{'penalty': ["l2"], 'dual': [True,False], "C": [0.1,1,10]},
      ##       {'penalty': ["l1"], 'dual': [False], "C": [0.1,1,10]},  ]
##

    # grid_params =  [# {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

    # grid_params = [   {'n_estimators': [  ], 'criterion': ['gini']},
    #   {'n_estimators': [10, 100, 1000], 'gamma': [0.0001], 'criterion': ['entropy']}, ]