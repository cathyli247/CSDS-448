import random
import time
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import scikitplot as skplt
import warnings
warnings.filterwarnings('ignore')


algorithms = {
    "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
    "RandomForest": ske.RandomForestClassifier(n_estimators=50),
    "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
    "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
    "GNB": GaussianNB(),
    "SVM": SVC(kernel='rbf', max_iter=100)
}

feature_num = [22, 34, 88]

def generate_data(permission_num):

    traindata = pd.read_csv('./' + str(permission_num) + '/train.csv')
    testdata = pd.read_csv('./' + str(permission_num) + '/newTest.csv')
    # traindata = traindata.sample(frac=1)
    # traindata.describe().to_csv('./' + str(permission_num) + '/data_stats.csv')
    return traindata, testdata

def plot_data(permission_num, algo):
    traindata, testdata = generate_data(permission_num)
    X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num)

    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)

    # Now plot.
    skplt.estimators.plot_feature_importances(clf,
                                              feature_names=traindata.columns.tolist(),
                                              x_tick_rotation=90,
                                              text_fontsize='small')

    plt.show()
    # skplt.metrics.plot_precision_recall_curve(y_test, probas)
    # plt.show()

def plot_learning_process(permission_num):
    traindata, testdata = generate_data(permission_num)
    fig, axs = plt.subplots(2,3, figsize=(12, 8), facecolor='w', edgecolor='k')
    X_train, X_test, y_train, y_test = get_train_and_test(traindata, testdata, permission_num)
    fig.tight_layout(pad=3.0)
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
        X = traindata.iloc[:,0:permission_num].head(train_size)
        Y = traindata.iloc[:,permission_num].head(train_size)
    C = testdata.iloc[:,permission_num]
    T = testdata.iloc[:,0:permission_num]

    traindata = np.array(X)
    trainlabel = np.array(Y)

    testdata = np.array(T)
    testlabel = np.array(C)

    return traindata, testdata, trainlabel, testlabel


def get_result(permission_num, size_list):
    result = {}
    traindata, testdata = generate_data(permission_num)

    for algo in algorithms:
        result[algo] = {'accuracy':[], 'precision':[], 'recall':[], 'fscore':[], 'runtime':[]}

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

    return result

def avg_report(result):
    data = {'accuracy':[], 'precision':[], 'recall':[], 'fscore':[], 'runtime':[]}
    table = {'accuracy':[], 'precision':[], 'recall':[], 'fscore':[], 'runtime':[]}
    for algo in algorithms:
        for key in result[algo].keys():
            data[key].append((algo,sum(result[algo][key])/len(result[algo][key])))
            table[key].append(sum(result[algo][key])/len(result[algo][key]))
    df = pd.DataFrame.from_dict(table, orient='index', columns=list(algorithms.keys()))
    print(df)


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
    for algo in algorithms:
        algo_result = result[algo]
        plt.plot(x, algo_result['runtime'], 'o-', label = algo)
    plt.title('Runtime')
    plt.xlabel('sample size')
    plt.legend(loc='upper left')
    plt.show()



n = np.linspace(0,3000, num=11).tolist()[1:]
result = get_result(88, n)
avg_report(result)
#
# plot_performance(result, n)

# plot_data(22, 'GradientBoosting')
# plot_learning_process(88)