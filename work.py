import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataset
def load_data(filepath):
    """
    Loads dataset from a CSV file.
    :param filepath: Path to the dataset CSV file.
    :return: Pandas DataFrame.
    """
    return pd.read_csv(filepath)


# Preprocess Dataset
def preprocess_data(data):
    """
    Handles preprocessing of the dataset.
    :param data: Pandas DataFrame.
    :return: Features (X), Target (y)
    """
    # Replace missing values marked as '?' with NaN
    data.replace('?', np.nan, inplace=True)

    # Drop irrelevant columns
    data.drop(columns=['encounter_id', 'patient_nbr'], inplace=True)

    # Fill missing values with mode for categorical and median for numerical columns
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Split features and target
    X = data.drop(columns=['readmitted'])
    y = data['readmitted']

    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Optimize Model
def optimize_model(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Best model from grid search.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data.
    :param model: Trained model.
    :param X_test: Test features.
    :param y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


# Main Execution
def main():
    # Filepath to the dataset (update with actual path)
    filepath = 'diabetic_data.csv'

    # Load and preprocess the data
    data = load_data(filepath)
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimize and train the model
    model = optimize_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
