# classifier.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def train_classifiers(X_train, y_train, X_test, y_test, X_valid, y_valid):
    # Initialize the classifiers
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(),
        'Linear SVM': LinearSVC(),
        'Random Forest': RandomForestClassifier()
    }

    # Train and evaluate the classifiers
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(f"{name} Classifier:")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print()

    # Select the best classifiers
    best_classifiers = [
        LogisticRegression(),
        RandomForestClassifier()
    ]

    # Perform hyperparameter tuning using GridSearchCV
    for classifier in best_classifiers:
        if isinstance(classifier, LogisticRegression):
            param_grid = {'C': [0.1, 1, 10]}
        elif isinstance(classifier, RandomForestClassifier):
            param_grid = {'n_estimators': [100, 200, 300]}

        grid_search = GridSearchCV(classifier, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {type(classifier).__name__}: {grid_search.best_params_}")
        print(f"Best score for {type(classifier).__name__}: {grid_search.best_score_}")
        print()

    # Train the final model
    final_model = LogisticRegression(C=grid_search.best_params_['C'])
    final_model.fit(X_train, y_train)

    # Save the final model
    pickle.dump(final_model, open('final_model.sav', 'wb'))

    return final_model