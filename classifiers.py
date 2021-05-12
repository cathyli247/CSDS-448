import random
import sys
import time
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import scikitplot as skplt
import warnings
warnings.filterwarnings('ignore')

algorithms = {
    "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
    "RandomForest": ske.RandomForestClassifier(n_estimators=50),
    "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
    "AdaBoost": ske.AdaBoostClassifier(n_estimators=50),
    "GNB": GaussianNB(),
    "SVM": SVC(kernel='rbf', max_iter=50,probability=True)
}

def generate_data(permission_num):
    traindata = pd.read_csv('./' + str(permission_num) + '/train.csv')
    testdata = pd.read_csv('./' + str(permission_num) + '/newTest.csv')
    # if permission_num == 88:
    # traindata = traindata.sample(frac=1)
    return traindata, testdata

def plot_data(permission_num, algo):
    traindata, testdata = generate_data(permission_num)
    X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num)

    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    skplt.estimators.plot_feature_importances(clf,
                                              figsize=(5, 7),
                                              feature_names=traindata.columns.tolist(),
                                              x_tick_rotation=90,
                                              text_fontsize='small')
    plt.show()

def plot_learning_process(permission_num, trainsize=None):
    traindata, testdata = generate_data(permission_num)
    fig, axs = plt.subplots(2,3, figsize=(12, 8), facecolor='w', edgecolor='k')
    X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num, trainsize)
    fig.tight_layout(pad=4.0)
    axs = axs.ravel()
    i = 0
    for algo in algorithms:
        clf = algorithms[algo]
        axs[i] = skplt.estimators.plot_learning_curve(clf, X_train, y_train, ax=axs[i])
        axs[i].set_title(algo)
        i += 1
    plt.show()

def get_train_and_test(traindata, testdata, permission_num,train_size=None):
    if train_size:
        X = traindata.iloc[:,0:permission_num].head(train_size)
        Y = traindata.iloc[:,permission_num].head(train_size)
    else:
        X = traindata.iloc[:,0:permission_num]
        Y = traindata.iloc[:,permission_num]
    C = testdata.iloc[:,permission_num]
    T = testdata.iloc[:,0:permission_num]

    traindata = np.array(X)
    trainlabel = np.array(Y)
    testdata = np.array(T)
    testlabel = np.array(C)

    return traindata, testdata, trainlabel, testlabel

def get_result(permission_num, size_list=None):
    result = {}
    pred = {}
    traindata, testdata = generate_data(permission_num)

    for algo in algorithms:
        result[algo] = {'accuracy':[], 'precision':[], 'recall':[], 'fscore':[], 'runtime':[], 'TN':[], 'FN':[], 'TP':[], 'FP':[]}

    if not size_list:
        size_list = [traindata.shape[1]]

    for n in size_list:
        n = int(n)
        X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num, n)
        for algo in algorithms:
            start_time = time.time()
            print("\nNow testing %s algorithms" % algo)
            clf = algorithms[algo]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            result[algo]['runtime'].append(time.time() - start_time)
            accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
            result[algo]['accuracy'].append(accuracy)
            report = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')
            result[algo]['precision'].append(report[0])
            result[algo]['recall'].append(report[1])
            result[algo]['fscore'].append(report[2])
            CM = sklearn.metrics.confusion_matrix(y_test, y_pred)
            result[algo]['TN'].append(CM[0][0])
            result[algo]['FN'].append(CM[1][0])
            result[algo]['TP'].append(CM[1][1])
            result[algo]['FP'].append(CM[0][1])
    return result

def avg_report(result):
    table = {}
    res = list(result.keys())[0]
    for key in result[res].keys():
        table[key] = []

    for algo in algorithms:
        for key in result[algo].keys():
            table[key].append(sum(result[algo][key])/len(result[algo][key]))
    df = pd.DataFrame.from_dict(table, orient='index', columns=list(algorithms.keys()))
    df.to_csv('result.csv')
    print(df)

    max_value = max(table.get('accuracy', ''))

    max_index = table.get('accuracy', '').index(max_value)

    return list(algorithms.keys())[max_index]


def plot_one_algo(ax, result, algo, stats, x):
    ax.plot(x, result[stats], 'o-', label=algo)
    ax.set_xlabel('sample size')
    ax.set_ylabel(stats)

def plot_performance(result, x):
    fig = plt.figure(figsize = (10,8))
    axes = fig.subplots(nrows=2, ncols=2)
    for algo in algorithms:
        algo_result = result[algo]
        plot_one_algo(axes[0,0], algo_result, algo, 'accuracy', x)
        plot_one_algo(axes[0,1], algo_result, algo, 'precision', x)
        plot_one_algo(axes[1,0], algo_result, algo, 'recall', x)
        plot_one_algo(axes[1,1], algo_result, algo, 'fscore', x)

    lines, labels = fig.axes[-1].get_legend_handles_labels()

    fig.legend(lines, labels,
               loc = 'upper right')
    plt.show()

def plot_runtime(result, x):
    for algo in algorithms:
        algo_result = result[algo]
        plt.plot(x, algo_result['runtime'], 'o-', label = algo)
    plt.title('Runtime')
    plt.xlabel('sample size')
    plt.legend(loc='upper left')
    plt.show()

def plot_auc(permission_num, n):
    traindata, testdata = generate_data(permission_num)
    X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num, n)
    for algo in algorithms:
        clf = algorithms[algo]
        scores = cross_val_score(estimator=clf,X=X_train,y = y_train, cv=5,scoring='roc_auc')
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), algo))

        y_pred = clf.fit(X_train,y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr,label='%s (auc = %0.2f)' % (algo, roc_auc))

    plt.legend(loc='lower right', prop={'size': 7})
    plt.plot([0, 1], [0, 1],linestyle='--',color='gray',linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.show()


def run(permssion_num, sample_size=3000):
    '''
    permission_num: choose from [22, 34, 88]
    '''

    if permssion_num not in [22, 34, 88]:
        permssion_num = 22

    n = np.linspace(0,sample_size, num=11).tolist()[1:]
    # n = [sample_size]
    result = get_result(22, n)
    # print training result
    final_clf = avg_report(result)

    # plot training performace
    plot_performance(result, n)
    plot_auc(permssion_num, n)
    plot_learning_process(permssion_num, sample_size)
    plot_runtime(result, n)

    majority_vote(permssion_num, n)

    # plot classification result
    plot_data(permssion_num, final_clf)

def majority_vote(permission_num, n):
    estimators = []
    for algo in algorithms.keys():
        t = (algo, algorithms[algo])
        estimators.append(t)

    eclf = VotingClassifier(estimators=estimators)
    traindata, testdata = generate_data(permission_num)
    X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num, n)
    eclf.fit(X_train, y_train)
    result = eclf.predict(X_test)
    print('number of malware: %s' % sum(result))
    print('number of benign: %s' % str(len(result) - sum(result)))
    print(eclf.score(X_test, y_test))


def main():
    sample_size = sys.argv[1]
    if not sample_size:
        sample_size = 3000
    if int(sample_size) > 30000:
        sample_size = 30000
    run(22, int(sample_size))

if __name__ == '__main__':
    main()

