from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os
import sys

def log(output_message, file = None):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    if file:
        file.write(str(output_message) + "\n")
    else:
        print(output_message)

def model_testing(model, X_columns, y_pred, y_test, logfile = None):
    acc_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    log(f"Model Accuracy: {acc_score: .4f}", logfile)
    log(f"Model Precision: {precision: .4f}", logfile)
    log(f"Model Recall: {recall: .4f}", logfile)
    log(f"Model F1-Score: {f1: .4f}\n", logfile)
    log(f"Classification Report:\n{class_report}", logfile)
    log(f"Confusion Matrix:\n{conf_matrix}\n", logfile)

    feature_importances = pd.DataFrame({
        'Feature': X_columns,
        'Importance': model.feature_importances_
    }).sort_values(by = 'Importance', ascending = False)
    log(f"Feature Importances:\n{feature_importances}", logfile)

def analyze_correlation(data, logfile = None):
    log("Correlation Analysis.\n", logfile)

    correlation_matrix = data.corr()['loan_paid_back'].sort_values(ascending = False)
    log(correlation_matrix, logfile)

def decision_tree(data, logfile = None):
    log("Decision Tree Model.\n", logfile)

    X = data.drop(columns = ['loan_paid_back'])
    y = data['loan_paid_back']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    ds_tree = DecisionTreeClassifier(random_state = 42, max_depth = 5)
    ds_tree.fit(X_train, y_train)
    y_pred = ds_tree.predict(X_test)

    model_testing(ds_tree, X.columns, y_pred, y_test, logfile)

def random_forest(data, logfile = None):
    log("Random Forest Model.\n", logfile)

    X = data.drop(columns = ['loan_paid_back'])
    y = data['loan_paid_back']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    rd_forest = RandomForestClassifier(n_estimators = 100, random_state = 42)
    rd_forest.fit(X_train, y_train)
    y_pred = rd_forest.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    model_testing(rd_forest, X.columns, y_pred, y_test, logfile)

def main():
    input_dir = "data/loan_dataset_cleaned_encoded.csv"
    data = pd.read_csv(input_dir)
    print(f"Data loaded from {input_dir} successfully.")
    print(f"Data Shape: {data.shape}")
    print(f"Data Columns: {data.columns.tolist()}\n")

    output_dir = "results/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "correlation.txt", "w", encoding = 'utf-8') as logfile:
        analyze_correlation(data, logfile)

    with open(output_dir + "decision_tree.txt", "w", encoding = 'utf-8') as logfile:
        decision_tree(data, logfile)
    
    with open(output_dir + "random_forest.txt", "w", encoding = 'utf-8') as logfile:
        random_forest(data, logfile)

if __name__ == "__main__":
    print(f"Python Version: {sys.version}")
    main()