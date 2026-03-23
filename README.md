# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load dataset, keep only label and message, encode ham=0 and spam=1, and drop missing values

2.Split data into features (X) and target (y), then into training and test sets.

3.Convert text messages to numerical features using TF-IDF vectorization.

4.Train a linear SVM classifier on the vectorized training data.

5.Predict on test data, evaluate accuracy, and classify new example messages as ham or spam.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Cassandra Suzanne F
RegisterNumber: 25014982

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the first two columns and rename them
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Convert labels: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove any rows with missing values
df = df.dropna()

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())

# Split data into features (X) and target (y)
X = df['message']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test with some example messages
test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE cruise to the Andaman! Call now to claim your prize!",
    "Can you pick up some milk on your way home?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately."
]

print("\n" + "="*50)
print("TESTING WITH EXAMPLE MESSAGES:")
print("="*50)

for msg in test_messages:
    msg_vectorized = vectorizer.transform([msg])
    prediction = svm_model.predict(msg_vectorized)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: {msg[:50]}... -> {result}")

*/
```

## Output:
<img width="953" height="302" alt="image" src="https://github.com/user-attachments/assets/51249b00-062f-477c-b715-ec034d5d83b6" />
<img width="401" height="212" alt="image" src="https://github.com/user-attachments/assets/34c681c1-9530-4118-b109-5155e8fbd4d8" />
<img width="844" height="291" alt="image" src="https://github.com/user-attachments/assets/bbd686b5-6194-4ac2-b59d-133a28a58368" />
<img width="1034" height="199" alt="image" src="https://github.com/user-attachments/assets/25698ed4-e4c9-445e-8eab-fcc702691f6d" />







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
