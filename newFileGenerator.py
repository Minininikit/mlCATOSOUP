import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier

default_data = pd.read_csv('csv/RING_V3_3.csv')
model = RandomForestClassifier(max_depth=2, min_samples_leaf=1, min_samples_split=9, n_estimators = 5000, random_state=42)
y = default_data['G']
data_features = ['A','B', 'C']
X = default_data[data_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_acc = f1_score(y_test,y_pred,average='weighted')
print(accuracy, f1_acc)

newData = pd.read_csv('csv/aboba_1.csv')
newX = newData[data_features]
G = model.predict(newX)
newData['G'] = G
newData.to_csv('output.csv', index=False)