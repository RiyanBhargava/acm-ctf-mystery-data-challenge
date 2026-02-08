import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('mystery_data.csv')

X = df[['feature_1', 'feature_2', 'feature_3']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, min_samples_splt=2, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model 13 Accuracy: {accuracy:.4f}")
print(f"flag{{{accuracy:.4f}}}")
print("\nIf this model doesn't work, try model_22.py next!")
