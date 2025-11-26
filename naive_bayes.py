import numpy as np
from collections import defaultdict


class GaussianNB_Univariate:
    def __init__(self):
        self.class_priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        labels = np.unique(y)
        for lab in labels:
            Xi = X[y == lab]
            self.class_priors[lab] = Xi.shape[0] / n
            self.means[lab] = Xi.mean(axis=0)
            self.vars[lab] = Xi.var(axis=0) + 1e-9

    def _gauss_logpdf(self, x, mean, var):
        return -0.5 * ((np.log(2 * np.pi * var)) + ((x - mean) ** 2) / var)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        labs = list(self.class_priors.keys())
        for i in range(X.shape[0]):
            scores = []
            xi = X[i]
            for lab in labs:
                logp = np.log(self.class_priors[lab])
                logp += self._gauss_logpdf(xi, self.means[lab], self.vars[lab]).sum()
                scores.append(logp)
            preds.append(labs[int(np.argmax(scores))])
        return np.array(preds)


class GaussianNB_Multivariate:
    def __init__(self):
        self.class_priors = {}
        self.means = {}
        self.covs = {}
        self.inv_covs = {}
        self.log_dets = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        labels = np.unique(y)
        for lab in labels:
            Xi = X[y == lab]
            self.class_priors[lab] = Xi.shape[0] / n
            self.means[lab] = Xi.mean(axis=0)
            cov = np.cov(Xi, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6
            self.covs[lab] = cov
            self.inv_covs[lab] = np.linalg.inv(cov)
            sign, ld = np.linalg.slogdet(cov)
            self.log_dets[lab] = ld

    def _logpdf(self, x, mean, inv_cov, log_det):
        d = x - mean
        return -0.5 * (d.dot(inv_cov).dot(d) + log_det + len(x) * np.log(2 * np.pi))

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        labs = list(self.class_priors.keys())
        for i in range(X.shape[0]):
            xi = X[i]
            scores = []
            for lab in labs:
                logp = np.log(self.class_priors[lab])
                logp += self._logpdf(xi, self.means[lab], self.inv_covs[lab], self.log_dets[lab])
                scores.append(logp)
            preds.append(labs[int(np.argmax(scores))])
        return np.array(preds)
