import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('mystery_data.csv')

X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MultinomialNB(alph=1.0)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model 30 Accuracy: {accuracy:.4f}")
print(f"flag{{{accuracy:.4f}}}")
print("\nIf this model doesn't work, try model_20.py next!")
