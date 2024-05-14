import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier


default_data = pd.read_csv('csv/RING_V2.csv')



#, max_depth, min_samples_leaf, min_samples_split, n_estimators

def Tr(max_depth, min_samples_leaf, min_samples_split, n_estimators):
    global model
    model = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators = n_estimators, random_state=42)
    


def dataPre(data):
    y = data['G']
    data_features = ['A','B', 'C']
    X = data[data_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    return X_train, X_test, y_train, y_test

def training(X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_acc = f1_score(y_test,y_pred,average='weighted')

    return accuracy, f1_acc

def validation(data):
    data_features2 = ['A','B', 'C']
    X_exam = data[data_features2]
    y_exam = data['G']
    
    y_pred = model.predict(X_exam)

    accuracy = accuracy_score(y_exam, y_pred)
    f1 = f1_score(y_exam,y_pred,average='weighted')

    return accuracy, f1, y_exam, y_pred