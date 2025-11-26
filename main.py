import argparse
import numpy as np
import time
from reader import read_arff
from utils import LabelEncoder, zscore_normalize
from cross_validation import stratified_k_fold
from metrics import accuracy_score, f1_score_macro, precision_score_macro
from knn import KNN
from perceptron import Perceptron
from mlp import MLP
from naive_bayes import GaussianNB_Univariate, GaussianNB_Multivariate
import csv

def run_experiment(path, folds=5, k_neighbors=5, hidden=32, epochs_perceptron=50, epochs_mlp=100):
    print('Reading ARFF...')
    attr_names, X_list, y_list = read_arff(path)
    print(f'Attributes: {len(attr_names)}  Instances: {len(X_list)}')

    X = np.array(X_list, dtype=float)
    le = LabelEncoder()
    y = np.array(le.fit_transform(y_list), dtype=int)
    n_classes = len(le.classes_)
    print('Classes:', le.classes_)

    methods = [
        ('KNN (Euclidiana)', 'knn_euc'),
        ('KNN (Manhattan)', 'knn_man'),
        ('Perceptron', 'perceptron'),
        ('MLP', 'mlp'),
        ('Naive Bayes (Univariado)', 'nb_uni'),
        ('Naive Bayes (Multivariado)', 'nb_multi')
    ]

    results = {}
    for _, key in methods:
        results[key] = {
            'acc': [], 'prec': [], 'f1': [],
            'train_time': [], 'test_time': []
        }

    fold_id = 0

    for train_idx, test_idx in stratified_k_fold(y.tolist(), folds):
        fold_id += 1
        print(f'Fold {fold_id}/{folds}')

        Xtr = X[train_idx]
        Xte = X[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        Xtr_n, Xte_n = zscore_normalize(Xtr, Xte)

        knn = KNN(k=k_neighbors, metric='euclidean')
        t0 = time.time()
        knn.fit(Xtr_n, ytr)
        t1 = time.time()
        ypred = knn.predict(Xte_n)
        t2 = time.time()

        results['knn_euc']['train_time'].append(t1 - t0)
        results['knn_euc']['test_time'].append(t2 - t1)
        results['knn_euc']['acc'].append(accuracy_score(yte, ypred))
        results['knn_euc']['prec'].append(precision_score_macro(yte, ypred))
        results['knn_euc']['f1'].append(f1_score_macro(yte, ypred))

        knn2 = KNN(k=k_neighbors, metric='manhattan')
        t0 = time.time()
        knn2.fit(Xtr_n, ytr)
        t1 = time.time()
        ypred = knn2.predict(Xte_n)
        t2 = time.time()

        results['knn_man']['train_time'].append(t1 - t0)
        results['knn_man']['test_time'].append(t2 - t1)
        results['knn_man']['acc'].append(accuracy_score(yte, ypred))
        results['knn_man']['prec'].append(precision_score_macro(yte, ypred))
        results['knn_man']['f1'].append(f1_score_macro(yte, ypred))

        perc = Perceptron(n_features=Xtr_n.shape[1], n_classes=n_classes, lr=0.1, epochs=epochs_perceptron)
        t0 = time.time()
        perc.fit(Xtr_n, ytr)
        t1 = time.time()
        ypred = perc.predict(Xte_n)
        t2 = time.time()

        results['perceptron']['train_time'].append(t1 - t0)
        results['perceptron']['test_time'].append(t2 - t1)
        results['perceptron']['acc'].append(accuracy_score(yte, ypred))
        results['perceptron']['prec'].append(precision_score_macro(yte, ypred))
        results['perceptron']['f1'].append(f1_score_macro(yte, ypred))

        mlp = MLP(n_features=Xtr_n.shape[1], n_hidden=hidden, n_classes=n_classes, lr=0.01, epochs=epochs_mlp)
        t0 = time.time()
        mlp.fit(Xtr_n, ytr)
        t1 = time.time()
        ypred = mlp.predict(Xte_n)
        t2 = time.time()

        results['mlp']['train_time'].append(t1 - t0)
        results['mlp']['test_time'].append(t2 - t1)
        results['mlp']['acc'].append(accuracy_score(yte, ypred))
        results['mlp']['prec'].append(precision_score_macro(yte, ypred))
        results['mlp']['f1'].append(f1_score_macro(yte, ypred))

        nb1 = GaussianNB_Univariate()
        t0 = time.time()
        nb1.fit(Xtr_n, ytr)
        t1 = time.time()
        ypred = nb1.predict(Xte_n)
        t2 = time.time()

        results['nb_uni']['train_time'].append(t1 - t0)
        results['nb_uni']['test_time'].append(t2 - t1)
        results['nb_uni']['acc'].append(accuracy_score(yte, ypred))
        results['nb_uni']['prec'].append(precision_score_macro(yte, ypred))
        results['nb_uni']['f1'].append(f1_score_macro(yte, ypred))

        nb2 = GaussianNB_Multivariate()
        t0 = time.time()
        nb2.fit(Xtr_n, ytr)
        t1 = time.time()
        ypred = nb2.predict(Xte_n)
        t2 = time.time()

        results['nb_multi']['train_time'].append(t1 - t0)
        results['nb_multi']['test_time'].append(t2 - t1)
        results['nb_multi']['acc'].append(accuracy_score(yte, ypred))
        results['nb_multi']['prec'].append(precision_score_macro(yte, ypred))
        results['nb_multi']['f1'].append(f1_score_macro(yte, ypred))

    summary = []

    for name, key in methods:
        acc = np.array(results[key]['acc'])
        prec = np.array(results[key]['prec'])
        f1 = np.array(results[key]['f1'])
        tr = np.array(results[key]['train_time'])
        ts = np.array(results[key]['test_time'])

        summary.append([
            name,
            f"{acc.mean():.4f} ± {acc.std():.4f}",
            f"{prec.mean():.4f} ± {prec.std():.4f}",
            f"{f1.mean():.4f} ± {f1.std():.4f}",
            f"{tr.mean():.4f} ± {tr.std():.4f}",
            f"{ts.mean():.4f} ± {ts.std():.4f}",
        ])

    with open("resultados_tabela.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Classificador", "Acurácia", "Precisão", "F1-Score", "Tempo Treino (s)", "Tempo Teste (s)"])
        w.writerows(summary)

    print("\nCSV 'resultados_tabela.csv' gerado com sucesso!\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs_perceptron', type=int, default=50)
    parser.add_argument('--epochs_mlp', type=int, default=100)
    args = parser.parse_args()

    run_experiment(args.data, args.folds, args.k, args.hidden, args.epochs_perceptron, args.epochs_mlp)
