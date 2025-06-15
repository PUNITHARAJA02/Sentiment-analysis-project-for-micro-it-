import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample data: you can replace this with a CSV file or dataset
data = {
    'text': [
        "I love this product!",
        "This is the worst movie ever.",
        "Amazing experience, I will come again.",
        "I hate this so much.",
        "It was okay, not great.",
        "Totally awesome and fun!",
        "Terrible, never buying again.",
        "I'm so happy with the service.",
        "This made me sad.",
        "Very satisfied with the quality!"
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Try your own prediction
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    print(f"Sentiment: {prediction[0]}")

# Example usage
predict_sentiment("I really enjoyed the show!")
predict_sentiment("It was a boring and dull experience.")
