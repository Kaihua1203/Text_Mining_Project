# Text Mining in Twitter Emotions

Based on [Twitter Emotions](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)

This is an assignment of Machine Learning and Data mining course. It mainly include the whole pipline of **muticlasses** text mining, from preprocessing data to ~~explaining model~~ evaluating model. The project implement the 3 wordembedding techniques(Bags of word, TF-IDF and Word2Vec), 5 different ML models(acctually 7), Topic Modelling and K-means.

Welcome everyone to commit it and correct my error in the code！

这是我在国外交换期间完成的一个关于多类别情感分类的数据挖掘课程大作业，项目涵盖了从数据预处理到模型评估的完整流程（模型解释未完成）。该项目其中包括3种不同文本转向量的处理方式，5种机器学习模型（实际上是七种，但两种运行时间较长），Topic Modelling 和 K-means。本人水平不高，代码有不足之处还请谅解。欢迎纠正我的错误。

## Installation

- Run 'git clone https://github.com/Kaihua1203/Text_Mining_Project.git' to create a copy of this git repository
- Run 'pip install -r requirements.txt' to install all package what you need. Some nltk parts might need to be downloaded manually. Just check the terminal.

## Usage

- EDA(Exploratory Data Analysis) and visualization is in jupyternote [Here](./EDA.ipynb)
- Run 'main.py'
- select the modes, sample size and ML model in the terminal. Everything can be executed on the terminal.
- ~~Model explanation~~

## Results

### Bags of words
- Logistic Regression
    - Accuracy: 0.872
    - Recall: 0.822
    - Precision: 0.829
    - AUC: 0.984
    - Running time: 16.180s
- Decision Tree
    - Accuracy: 0.812
    - Recall: 0.752
    - Precision: 0.751
    - AUC: 0.871
    - Running time: 155.094s
- XGB(XGBoosting)
    - Accuracy: 0.874
    - Recall: 0.859
    - Precision: 0.829
    - AUC: 0.986
    - Running time: 17.650s
- LGB(Light Gradient Boosting Machine)
    - Accuracy: 0.886
    - Precision: 0.835
    - AUC: 0.990
    - Running time: 12.966s
- MNB(MultinomialNB)
    - Accuracy: 0.837
    - Recall: 0.720
    - Precision: 0.834
    - AUC: 0.968
    - Running time: 0.111s

### Bags of Word(without Stemming and Lemmatization)

- Logistic Regression
    - Accuracy: 0.892
    - Recall: 0.849
    - Precision: 0.845
    - AUC: 0.991
    - Running time: 18.084s
- Decision Tree
    - Accuracy: 0.853
    - Recall: 0.796
    - Precision: 0.796
    - AUC: 0.897
    - Running time: 131.828s
- XGB(XGBoosting)
    - Accuracy: 0.897
    - Recall: 0.899
    - Precision: 0.850
    - AUC: 0.991
    - Running time: 22.991s
- LGB(Light Gradient Boosting Machine)
    - Accuracy: 0.908
    - Precision: 0.903
    - AUC: 0.995
    - Running time: 16.758s
- MNB(MultinomialNB)
    - Accuracy: 0.866
    - Recall: 0.756
    - Precision: 0.866
    - AUC: 0.977
    - Running time: 0.119s

### TF-IDF
- Logistic Regression
    - Accuracy: 0.875
    - Recall: 0.816
    - Precision: 0.841
    - AUC: 0.986
    - Running time: 16.196s
- Decision Tree
    - Accuracy: 0.803
    - Recall: 0.738
    - Precision: 0.742
    - AUC: 0.866
    - Running time: 182.401s
- XGB(XGBoosting)
    - Accuracy: 0.872
    - Recall: 0.852
    - Precision: 0.829
    - AUC: 0.985
    - Running time: 143.959s
- LGB(Light Gradient Boosting Machine)
    - Accuracy: 0.884
    - Precision: 0.834
    - AUC: 0.990
    - Running time: 44.493s
- MNB(MultinomialNB)
    - Accuracy: 0.750
    - Recall: 0.538
    - Precision: 0.864
    - AUC: 0.966
    - Running time: 0.113s

### Word2Vec
- Logistic Regression
    - Accuracy: 0.579
    - Recall: 0.441
    - Precision: 0.609
    - AUC: 0.822
    - Running time: 24.605s
- Decision Tree
    - Accuracy: 0.358
    - Recall: 0.263
    - Precision: 0.264
    - AUC: 0.566
    - Running time: 149.365s
- XGB(XGBoosting)
    - Accuracy: 0.586
    - Recall: 0.450
    - Precision: 0.593
    - AUC: 0.834
    - Running time: 82.818s
- LGB(Light Gradient Boosting Machine)
    - Accuracy: 0.575
    - Recall: 0.427
    - Precision: 0.618
    - AUC: 0.826
    - Running time: 66.111s
- MNB(MultinomialNB)
    - Accuracy: 0.353
    - Recall: 0.176
    - Precision: 0.139
    - AUC: 0.719
    - Running time: 1.186s

### Topic Modeling

    Topic 1:
    love im like want know hate life everyth friend make famili time alon happi
    Topic 2:
    help peopl way make life think read mani thing time know sad word hope
    Topic 3:
    im pretti littl today href like realli bit morn quit right good night run
    Topic 4:
    like realli look peopl want think make accept know say someth dont person need
    Topic 5:
    time like im day dont littl know ive bit thing work realli start week
    Topic 6:
    like ive support bodi wonder start year felt hand ach walk miss disgust day

Visualization see [here](./images/Topic%20Modeling.png)


## Issues

- Clustering(Kmeans) seems to behave weirdly for now. Unsure if this is normal or if the code is wrong.

## Observations

- SVM and Random Forest are time consuming in large dataset. 
- Word vector is a sparse matrix, some ML models need dense matrix. If using 'numpy.toarray()', it will takes another
a long time to run.
- Since LightGBM expects input features to be in 'float32/64' format, the labels should be converted from 'int' to 'float32/64'.
- After Word2Vec technique transfers, word vectors have negative values. Since negative values will be passed in MultinomialNB, the data should be scaled to [0, 1] by 'MinMaxScaler()'.
- In muticlass classfication, AUC scores can't be calculate by 'roc_auc_score(y_test, y_pred)' due to definition of ROC. So I use 'roc_auc_score(y_test, y_pred_proba, multi_class='ovr')', involving calculating the binary AUC for each class versus the rest and then averaging these AUC values.
- That's interesting to see the model without Stemming and Lemmatization has the better performance than the one with this preprocessing method:)

## TODO
- Explain the model by LIME or SHAP