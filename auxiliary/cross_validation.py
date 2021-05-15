
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error¶


approach='svd'
scoring = sklearn.metrics.mean_squared_error¶


def perform_crossvalidation(approach, data, labels, cv=5, scoring=):


