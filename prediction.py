# prediction.py

import pickle
from DataPrep import preprocess_data
from FeatureSelection import select_features
from classifier import train_classifiers

def predict_fake_news(news_headline, model, tfidf_vectorizer):
    # Preprocess the input news headline
    news_headline_tfidf = tfidf_vectorizer.transform([news_headline])

    # Make predictions
    prediction = model.predict(news_headline_tfidf)
    probability = model.predict_proba(news_headline_tfidf)

    return prediction[0], probability[0]

def main():
    # Preprocess the data
    X_train_tfidf, y_train, X_test_tfidf, y_test, X_valid_tfidf, y_valid, tfidf_vectorizer = preprocess_data()

    # Select features
    X_train_selected, X_test_selected, X_valid_selected = select_features(X_train_tfidf, y_train, X_test_tfidf, X_valid_tfidf)

    # Train the classifiers and get the final model
    final_model = train_classifiers(X_train_selected, y_train, X_test_selected, y_test, X_valid_selected, y_valid)

    # Save the final model and vectorizer
    pickle.dump(final_model, open('final_model.sav', 'wb'))
    pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.sav', 'wb'))

    # Get user input for news headline
    news_headline = input("Enter a news headline: ")

    # Load the trained model and vectorizer
    loaded_model = pickle.load(open('final_model.sav', 'rb'))
    loaded_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

    # Make predictions
    prediction, probability = predict_fake_news(news_headline, loaded_model, loaded_vectorizer)
    print(f"Prediction: {prediction}")
    print(f"Probability of truth: {probability[1]}")

if __name__ == '__main__':
    main()