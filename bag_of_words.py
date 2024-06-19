from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from result_visualization import *
import numpy as np
import time

def run_bag_of_words(train_text, train_label, test_text, test_label, models):
    # bags of word
    vectorizer = CountVectorizer(dtype=np.float64)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    
    metrics = {'model': [], 'accuracy': [], 'recall': [], 'precision': []}

    if 'LogR' in models or 'all' in models:
        # Logistic Regression model
        start_time = time.time()

        # metric
        logR_model = LogisticRegression()
        logR_model.fit(X_train, train_label)
        logR_pred = logR_model.predict(X_test)
        logR_pred_proba = logR_model.predict_proba(X_test)

        end_time = time.time()
        
        accuracy_log = accuracy_score(test_label, logR_pred)
        recall_log = recall_score(test_label, logR_pred, average='macro')
        precision_log = precision_score(test_label, logR_pred, average='macro')
        auc_log = roc_auc_score(test_label, logR_pred_proba, multi_class='ovr')
        metrics['model'].append('LogR')
        metrics['accuracy'].append(accuracy_log)
        metrics['recall'].append(recall_log)
        metrics['precision'].append(precision_log)

        print('Logistic Regression (Bag of Words) Accuracy: {:.3f}'.format(accuracy_log))
        print('Recall: {:.3f}'.format(recall_log))
        print('Precision: {:.3f}'.format(precision_log))
        print('AUC: {:.3f}'.format(auc_log))
        print('Running time: {:.3f}s'.format(end_time - start_time))

        run_confusion_matrix(test_label, logR_pred, 
                             label_name=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
                             title="Confusion Matrix of Logistic Regression based on BoW",
                             save_path="./images/Confusion_Matrix_of_LogR_ON_BoW2.png")

    if 'DT' in models or 'all' in models:
        # Linear Regression model
        start_time = time.time()

        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, train_label)
        dt_pred = dt_model.predict(X_test)
        dt_pred_proba = dt_model.predict_proba(X_test)

        end_time = time.time()

        accuracy_dt = accuracy_score(test_label, dt_pred)
        recall_dt = recall_score(test_label, dt_pred, average='macro')
        precision_dt = precision_score(test_label, dt_pred, average='macro')
        auc_dt = roc_auc_score(test_label, dt_pred_proba, multi_class='ovr')
        metrics['model'].append('DT')
        metrics['accuracy'].append(accuracy_dt)
        metrics['recall'].append(recall_dt)
        metrics['precision'].append(precision_dt)

        print('Decision Tree (Bag of Words) Accuracy: {:.3f}'.format(accuracy_dt))
        print('Recall: {:.3f}'.format(recall_dt))
        print('Precision: {:.3f}'.format(precision_dt))
        print('AUC: {:.3f}'.format(auc_dt))
        print('Running time: {:.3f}s'.format(end_time - start_time))

        run_confusion_matrix(test_label, dt_pred, 
                             label_name=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
                             title="Confusion Matrix of Decision Tree based on BoW",
                             save_path="./images/Confusion_Matrix_of_DT_ON_BoW2.png")

    if 'XGB' in models or 'all' in models:
        # XGB model
        start_time = time.time()

        xgb_model = XGBClassifier(probability=True)
        xgb_model.fit(X_train, train_label)
        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)
        
        end_time = time.time()

        accuracy_xgb = accuracy_score(test_label, xgb_pred)
        recall_xgb = recall_score(test_label, xgb_pred, average='macro')
        precision_xgb = precision_score(test_label, xgb_pred, average='macro')
        auc_xgb = roc_auc_score(test_label, xgb_pred_proba, multi_class='ovr')
        metrics['model'].append('XGB')
        metrics['accuracy'].append(accuracy_xgb)
        metrics['recall'].append(recall_xgb)
        metrics['precision'].append(precision_xgb)

        print('XGB (Bag of Words) Accuracy: {:.3f}'.format(accuracy_xgb))
        print('Recall: {:.3f}'.format(recall_xgb))
        print('Precision: {:.3f}'.format(precision_xgb))
        print('AUC: {:.3f}'.format(auc_xgb))
        print('Running time: {:.3f}s'.format(end_time - start_time))

        run_confusion_matrix(test_label, xgb_pred, 
                             label_name=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
                             title="Confusion Matrix of XGB based on BoW",
                             save_path="./images/Confusion_Matrix_of_XGB_ON_BoW2.png")

    if 'LGB' in models or 'all' in models:
        # LGBM model
        start_time = time.time()

        lgb_model = LGBMClassifier()
        lgb_model.fit(X_train, train_label)
        lgb_pred = lgb_model.predict(X_test)
        lgb_pred_proba = lgb_model.predict_proba(X_test)

        end_time = time.time()

        accuracy_lgb = accuracy_score(test_label, lgb_pred)
        recall_lgb = recall_score(test_label, lgb_pred, average='macro')
        precision_lgb = precision_score(test_label, lgb_pred, average='macro')
        auc_lgb = roc_auc_score(test_label, lgb_pred_proba, multi_class='ovr')
        metrics['model'].append('LGB')
        metrics['accuracy'].append(accuracy_lgb)
        metrics['recall'].append(recall_lgb)
        metrics['precision'].append(precision_lgb)

        print('LGB (Bag of Words) Accuracy: {:.3f}'.format(accuracy_lgb))
        print('Recall: {:.3f}'.format(recall_lgb))
        print('Precision: {:.3f}'.format(precision_lgb))
        print('AUC: {:.3f}'.format(auc_lgb))
        print('Running time: {:.3f}s'.format(end_time - start_time))

        run_confusion_matrix(test_label, lgb_pred, 
                             label_name=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
                             title="Confusion Matrix of LGB based on BoW",
                             save_path="./images/Confusion_Matrix_of_LGB_ON_BoW2.png")


    if 'MNB' in models or 'all' in models:
        # MultinomialNB model
        start_time = time.time()

        mnb_model = MultinomialNB()
        mnb_model.fit(X_train, train_label)
        mnb_pred = mnb_model.predict(X_test)
        mnb_pred_proba = mnb_model.predict_proba(X_test)
        
        end_time = time.time()

        accuracy_mnb = accuracy_score(test_label, mnb_pred)
        recall_mnb = recall_score(test_label, mnb_pred, average='macro')
        precision_mnb = precision_score(test_label, mnb_pred, average='macro')
        auc_mnb = roc_auc_score(test_label, mnb_pred_proba, multi_class='ovr')
        metrics['model'].append('MNB')
        metrics['accuracy'].append(accuracy_mnb)
        metrics['recall'].append(recall_mnb)
        metrics['precision'].append(precision_mnb)

        print('MNB (Bag of Words) Accuracy: {:.3f}'.format(accuracy_mnb))
        print('Recall: {:.3f}'.format(recall_mnb))
        print('Precision: {:.3f}'.format(precision_mnb))
        print('AUC: {:.3f}'.format(auc_mnb))
        print('Running time: {:.3f}s'.format(end_time - start_time))

        run_confusion_matrix(test_label, mnb_pred, 
                             label_name=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
                             title="Confusion Matrix of MNB based on BoW",
                             save_path="./images/Confusion_Matrix_of_MNB_ON_BoW2.png")
    
    run_plot_metrics(metrics, save_path="./images/Comparison of Metrics in BoW2.png")
