from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    evaluation_metrics = {
        'Confusion_Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    return evaluation_metrics

def decision_tree_model(df, dep_variable ,test_size=0.2, random_state=42):
    # scaler = MinMaxScaler()
    # df['Age of Mother'] = scaler.fit_transform(df[['Age of Mother']]) NOT needed as decision trees

    X = df.drop(columns=[dep_variable])
    y = df[dep_variable]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the decision tree classifier
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)
    y_pred = decision_tree_model.predict(X_test)
    evaluation_metrics = evaluate_model(y_test, y_pred)

    return decision_tree_model, evaluation_metrics
df=pd.read_csv("data/FINAL_DATA_FOR_REG.csv")
df.dropna(inplace=True)

# decision_tree_model, evaluation_metrics=decision_tree_model(df,"Hospital delivery")
# print(evaluation_metrics)
# {'Confusion_Matrix': array([[175,   0],
#        [  1, 189]]), 'Accuracy': 0.9972602739726028, 'Precision': 0.9972758405977584, 'Recall': 0.9972602739726028, 'F1-Score': 0.9972605623019475}
# decision_tree_model, evaluation_metrics=decision_tree_model(df,"PrenatalCare")
# print(evaluation_metrics)
# {'Confusion_Matrix': array([[365]]), 'Accuracy': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'F1-Score': 1.0}
decision_tree_model, evaluation_metrics=decision_tree_model(df,"Child is alive")
print(evaluation_metrics)
# {'Confusion_Matrix': array([[  1,   8],
#        [ 15, 341]]), 'Accuracy': 0.936986301369863, 'Precision': 0.9545261412254191, 'Recall': 0.936986301369863, 'F1-Score': 0.9454953852132518}