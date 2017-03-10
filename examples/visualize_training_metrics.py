# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from pydd.MLP import MLPfromArray
from sklearn import datasets, metrics, model_selection, preprocessing


# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility

# Dataset parameters
n_samples = 10000
n_classes = 2
test_size = 0.2

# Model parameters
params = {'port': 8085, 'nclasses': n_classes, 'gpu': True, 'gpuid': 0}
fit_params = {'iterations': 500, 'display_metric_interval': 1., 'base_lr': 0.01, 'test_interval': 100}

X, y = datasets.make_classification(n_samples=n_samples, n_classes=n_classes, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)


clf = MLPfromArray(**params)
logs = clf.fit(x_train, y_train, **fit_params)

y_test_prob = clf.predict_proba(x_test)
y_test_pred = y_test_prob.argmax(-1)

report = metrics.classification_report(y_test, y_test_pred)

print(report)

