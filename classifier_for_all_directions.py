from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset = pd.read_csv("./final_dataset_for_all_directions.csv")

dataset = dataset.drop(['Unnamed: 0', 'time'], axis=1)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbc = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1, max_depth=5)

for i in tqdm(range(gbc.n_estimators), desc="Training Gradient Boosting"):
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy is {accuracy:.3f}")