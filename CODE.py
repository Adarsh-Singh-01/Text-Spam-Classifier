import pandas as pd

# Loading dataset
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])
print(df.head())
print(df['label'].value_counts())




import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Simple text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Classifier Training
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Prediction
y_pred = model.predict(X_test_vectors)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))




