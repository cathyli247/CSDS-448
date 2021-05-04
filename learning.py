import random
import time
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
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

def generate_data(sample_size=None):
    data = pd.read_csv('data.csv', sep='|')
    data = data.sample(frac = 1)

    if sample_size:
        sample_size = int(sample_size)
        X = data.drop(['Name', 'md5', 'legitimate'], axis=1).head(sample_size).values
        y = data['legitimate'].head(sample_size).values
    else:
        X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
        y = data['legitimate'].values
    # print('Researching important feature based on %i total features\n' % X.shape[1])
    #
    fsel = ske.ExtraTreesClassifier().fit(X, y)
    model = SelectFromModel(fsel, prefit=True)
    X_new = model.transform(X)
    # nb_features = X_new.shape[1]
    # features = []
    #
    # print('%i features identified as important:' % nb_features)
    #
    # indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
    # for f in range(nb_features):
    #     print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

    # XXX : take care of the feature order
    # for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    #     features.append(data.columns[2+f])
    return X_new, y


def get_result(n=None):
    # X_new, y = generate_data()
    result = {}

    for algo in algorithms:
        result[algo] = {'accuracy':[], 'precision':[], 'recall':[], 'fscore':[], 'runtime':[]}

    for i in n:
        X_new, y = generate_data(i)
        print('Iteration:', i)
        X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.2)
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
# print(n)
# n = [100, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
result = get_result(n)
avg_report(result)

plot_performance(result, n)