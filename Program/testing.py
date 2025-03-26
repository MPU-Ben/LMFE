import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    roc_curve, f1_score, recall_score, matthews_corrcoef, auc, precision_recall_curve
def evaluate(args):

    ### Test Dataset ###
    splited = args['splited']
    subfold = args['subfold']
    D = pd.read_csv(args['testDatasetPath'])
    all_columns = D.columns.tolist()
    label_name = 'label'
    feature_columns = [col for col in all_columns if col != label_name]
    feature_names = feature_columns
    X_test = D.iloc[:,:-1].values
    y_test = D.iloc[:, -1].values
    scale = StandardScaler()
    X_test = scale.fit_transform(X_test)

    metrics = testModel(X_test, y_test,subfold, feature_names)
    return metrics

def testModel(X_test, y_test, subfold,feature_names):

    import joblib
    with open('../Files/'+subfold+'/dumpModel.pkl', 'rb') as File:
        model = joblib.load(File)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_artificial = model.predict(X_test)


    CM = confusion_matrix(y_pred=y_artificial, y_true=y_test)
    TN, FP, FN, TP = CM.ravel()

    accuracy = "{:.3f}".format(accuracy_score(y_pred=y_artificial, y_true=y_test) * 100.0)
    precision = "{:.3f}".format(precision_score(y_true=y_test, y_pred=y_artificial) * 100)
    recall = "{:.3f}".format(recall_score(y_true=y_test, y_pred=y_artificial) * 100)
    auROC = "{:.3f}".format(roc_auc_score(y_true=y_test, y_score=y_proba))
    auPR = "{:.3f}".format(average_precision_score(y_true=y_test, y_score=y_proba))
    f1 = "{:.3f}".format(f1_score(y_true=y_test, y_pred=y_artificial))
    mcc = "{:.3f}".format(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
    sensitivity = "{:.3f}".format((TP / (TP + FN)) * 100.0)
    specificity = "{:.3f}".format((TN / (TN + FP)) * 100.0)

    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'auROC', 'auPR', 'F1 Score', 'MCC', 'Sensitivity', 'Specificity'],
        'Value': [accuracy, precision, recall, auROC, auPR, f1, mcc, sensitivity, specificity]
    })
    results_transposed_df = results_df.round(2).transpose()
    print(results_transposed_df)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'auROC': auROC,
        'auPR': auPR,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
def evaluateByOtherSpeice(args):
    D = pd.read_csv(args['AllDataset'])
    all_columns = D.columns.tolist()
    label_name = 'label'
    feature_columns = [col for col in all_columns if col != label_name]
    feature_names = feature_columns
    specieName = args['specieName']
    subfold = args['subfold']
    X_otherspeice = D.iloc[:, :-1].values
    y_otherspeice = D.iloc[:, -1].values
    # # process with unblanced data
    # smote = SMOTE(random_state=42)
    # X_otherspeice, y_otherspeice = smote.fit_resample(X_otherspeice, y_otherspeice)

    scale_otherspeice = StandardScaler()
    X_otherspeice_=scale_otherspeice.fit_transform(X_otherspeice)
    evalModel(X_otherspeice_, y_otherspeice, specieName, subfold, feature_names)

roc_aucs = []
fprs = []
tprs = []
precisions = []
recalls = []
specieNames = []
def evalModel(X, y, specieName, subfold, feature_names):
    import joblib
    with open('../Files/'+subfold+'/dumpModel.pkl', 'rb') as File:
        model = joblib.load(File)
    y_proba = model.predict_proba(X)[:, 1]
    y_artificial = model.predict(X)
    CM = confusion_matrix(y_pred=y_artificial, y_true=y)
    labels = ['lncRNA','mRNA']  # 根据您的具体类别设置
    # plot_confusion_matrix(CM, labels, subfold, specieName)
    TN, FP, FN, TP = CM.ravel()

    accuracy = "{:.3f}".format(accuracy_score(y_pred=y_artificial, y_true=y) * 100.0)
    precision = "{:.3f}".format(precision_score(y_true=y, y_pred=y_artificial) * 100)
    recall = "{:.3f}".format(recall_score(y_true=y, y_pred=y_artificial) * 100)
    auROC = "{:.3f}".format(roc_auc_score(y_true=y, y_score=y_proba))
    auPR = "{:.3f}".format(average_precision_score(y_true=y, y_score=y_proba))
    f1 = "{:.3f}".format(f1_score(y_true=y, y_pred=y_artificial))
    mcc = "{:.3f}".format(matthews_corrcoef(y_true=y, y_pred=y_artificial))
    sensitivity = "{:.3f}".format((TP / (TP + FN)) * 100.0)
    specificity = "{:.3f}".format((TN / (TN + FP)) * 100.0)

    results_df = pd.DataFrame({
        'Metric': ['specieName','Accuracy', 'Precision', 'Recall', 'auROC', 'auPR', 'F1-Score', 'MCC', 'Sensitivity', 'Specificity'],
        'Value': [specieName, accuracy, precision, recall, auROC, auPR, f1, mcc, sensitivity, specificity]
    })


    results_transposed_df = results_df.round(2).transpose()
    print(results_transposed_df)

    fpr, tpr, thresholds = roc_curve(y, y_proba)

    fprs.append(fpr)
    tprs.append(tpr)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    specieNames.append(specieName)
    precisions.append(precision)
    recalls.append(recall)
    if y_proba.ndim == 1:
        non_coding_probability =  y_proba
        coding_probability = 1 - y_proba
    else:
        non_coding_probability = y_proba[:, 0]
        coding_probability = y_proba[:, 1]
    noncoding_probability = [f"{p:.8f}" for p in non_coding_probability]
    coding_probability = [f"{p:.8f}" for p in coding_probability]
    true_label = ['noncoding' if label == 1 else 'coding' for label in y]
    prediction_label = ['noncoding' if label == 1 else 'coding' for label in y_artificial]
    output_df = pd.DataFrame({
        'true_label': true_label,
        'prediction_label': prediction_label,
        'non-coding_probability': noncoding_probability,
        'coding_probability': coding_probability
    })

    output_df.to_csv('../Files/' + subfold + '/LMFE_' + specieName + '_predictions.csv', index=False)
