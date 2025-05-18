import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def load_preprocessed_data(file_path):
    """Load preprocessed data from file."""
    df = pd.read_csv(file_path)
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y

def train_and_evaluate(X, y):
    """Train and evaluate the model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    return model, scaler

def save_model(model, scaler, model_path, scaler_path):
    """Save the trained model and scaler."""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Load preprocessed data
    X, y = load_preprocessed_data('data/preprocessed_data.csv')
    
    # Train and evaluate the model
    model, scaler = train_and_evaluate(X, y)
    
    # Save the model and scaler
    save_model(model, scaler, 'models/ipl_predictor.joblib', 'models/scaler.joblib') 