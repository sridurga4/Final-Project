# FeatureSelection.py

from sklearn.feature_selection import SelectKBest, chi2


def select_features(X_train, y_train, X_test, X_valid, k=5000):
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    X_valid_selected = selector.transform(X_valid)

    return X_train_selected, X_test_selected, X_valid_selected