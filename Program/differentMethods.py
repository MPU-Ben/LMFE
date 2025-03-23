# Avoiding warnings
import warnings
import time
def warn(*args, **kwargs): pass
warnings.warn = warn

# Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score, recall_score,
    matthews_corrcoef,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier

# Classifier Names
Names = ['KNN', 'DT', 'NB', 'SVM', 'BG', 'RF', 'AB', 'GBDT', 'XGBOOST']
Classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(probability=True),
    BaggingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
]


def optimalParameter(model, X, y):
    print('Start optimizing model:', model.__class__.__name__)
    param_grid = {}

    # Define parameter grids for each model
    if isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'max_iter': [100, 300, 500, 700],
            'solver': ['newton-cg', 'liblinear', 'sag', 'saga']
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    elif isinstance(model, DecisionTreeClassifier):
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(model, GaussianNB):
        param_grid = {
            'priors': [None, [0.25, 0.25, 0.5], [0.5, 0.25, 0.25]],
            'var_smoothing': [1e-9, 1e-6, 1e-3]
        }
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.1, 1.0, 10.0],
            'degree': [2, 3, 4]
        }
    # Add additional classifiers here...

    if param_grid:
        best_params, best_score = showOptimalResult(X, y, param_grid, model)
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
    print('End optimization.')

def showOptimalResult(X, y, param_space, model):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42
    )

    random_search.fit(X, y)
    best_params = random_search.best_params_
    best_score = -random_search.best_score_

    print("Best Parameters (Random Search):", best_params)
    print("Best Score (Random Search):", abs(best_score))

    # Fine-tune using GridSearchCV
    param_space_fine = {key: [best_params[key]] for key in best_params.keys()}
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_space_fine,
        scoring='neg_mean_squared_error',
        cv=5,
    )
    grid_search.fit(X, y)
    grid_best_params = grid_search.best_params_
    grid_best_score = -grid_search.best_score_

    print("Best Parameters (Grid Search):", grid_best_params)
    print("Best Score (Grid Search):", -grid_best_score)
    return grid_best_params, grid_best_score

def runDifferentMethods(args):
    D = pd.read_csv(args['dataset'])
    X = D.iloc[:, :-1].values
    y = D.iloc[:, -1].values

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Results, accMeans, training_times = [], [], []
    cv = StratifiedKFold(n_splits=args['nFCV'], shuffle=True)

    for classifier, name in zip(Classifiers, Names):
        accuracies, auROC, avePrecision, F1_Score, MCC, Recall = [], [], [], [], [], []
        CM = np.zeros((2, 2), dtype=int)
        print('{} is starting.'.format(name))
        start_time = time.time()

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Scale the features for each fold
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1]

            accuracies.append(accuracy_score(y_test, y_pred))
            auROC.append(roc_auc_score(y_test, y_proba))
            avePrecision.append(average_precision_score(y_test, y_proba))
            F1_Score.append(f1_score(y_test, y_pred))
            MCC.append(matthews_corrcoef(y_test, y_pred))
            Recall.append(recall_score(y_test, y_pred))
            CM += confusion_matrix(y_test, y_pred)

        training_time = time.time() - start_time
        print(f'{name} training time: {training_time:.2f} seconds')

        Results.append([accuracy * 100 for accuracy in accuracies])
        accMeans.append(np.mean(accuracies))

        TN, FP, FN, TP = CM.ravel()
        metrics = {
            'Metric': ['Accuracy', 'auROC', 'auPR', 'F1_Score', 'MCC', 'Recall', 'Sensitivity', 'Specificity'],
            'Value': [
                np.mean(accuracies) * 100,
                np.mean(auROC),
                np.mean(avePrecision),
                np.mean(F1_Score),
                np.mean(MCC),
                np.mean(Recall),
                (TP / (TP + FN)) * 100.0,
                (TN / (TN + FP)) * 100.0
            ],
        }

        # Create a DataFrame from the dictionary
        metrics_df = pd.DataFrame(metrics)

        # Add confusion matrix as a separate row
        confusion_matrix_row = pd.DataFrame({
            'Metric': ['Confusion Matrix'],
            'Value': [str(CM)]
        })
        metrics_df = pd.concat([metrics_df, confusion_matrix_row], ignore_index=True)
        # Save the DataFrame to a CSV file
        metrics_df.to_csv('../Files/evaluationMetrics.csv', index=False)
        import joblib
        joblib.dump(classifier, f'../Files/models/{name}_dumpModel.pkl')

    # Plotting functions
    if args.get('auROC', 0) == 1:
        auROCplot()
    if args.get('boxPlot', 0) == 1:
        boxPlot(Results, Names)
    if args.get('accPlot', 0) == 1:
        accPlot(accMeans, Names)
    if args.get('timePlot', 0) == 1:
        plot_training_times(training_times, Names)


def plot_training_times(training_times, Names):
    plt.figure(figsize=(12, 8))
    plt.bar(Names, training_times)
    plt.xlabel('Model', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.ylabel('Training Time (seconds)', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.savefig('../Files/Time_consume.png', dpi=300)
    plt.show()

def boxPlot(Results, Names):
    plt.figure(figsize=(9, 8))
    plt.boxplot(Results, patch_artist=True, vert=True)
    plt.xticks(range(1, len(Names) + 1), Names, rotation=45)
    plt.xlabel('Classifiers', fontdict={'fontsize': 13, 'fontweight': 'bold'})
    plt.ylabel('Accuracy (%)', fontdict={'fontsize': 13, 'fontweight': 'bold'})
    plt.grid(axis='y')
    plt.savefig('../Files/Accuracy_boxPlot.png', dpi=300)
    plt.show()

def accPlot(Results, Names):
    plt.figure()
    plt.plot(Names, Results, lw=2, color='g')
    plt.scatter(Names, Results, color='r', marker='o')
    plt.xlabel('Classifier', fontdict={'fontsize': 13, 'fontweight': 'bold'})
    plt.ylabel('Accuracy (%)', fontdict={'fontsize': 13, 'fontweight': 'bold'})
    plt.grid(True)
    plt.savefig('../Files/acc.png', dpi=300)
    plt.show()

def auROCplot():
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontdict={'fontsize': 13, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate (TPR)', fontdict={'fontsize': 13, 'fontweight': 'bold'})
    plt.legend(loc='lower right')
    plt.savefig('../Files/auROC.png', dpi=300)
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.05, 1., 20), verbose=0):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                     train_scores_mean + np.std(train_scores, axis=1), alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                     test_scores_mean + np.std(test_scores, axis=1), alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()