import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load the IPL match data."""
    df = pd.read_csv(file_path)
    
    # Print sample of raw data
    print("\nSample of raw data:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    return df

def convert_overs_to_balls(over):
    """Convert over number (e.g., 4.3) to total balls (e.g., 27)."""
    try:
        main_over = int(float(over))
        decimal = float(over) % 1
        balls = int(round(decimal * 10))  # Convert .1, .2, etc. to actual ball numbers
        return main_over * 6 + balls
    except:
        return 0

def calculate_match_result(group):
    """Calculate if the batting team won the match."""
    final_score = group['runs'].sum()
    target = group['total'].iloc[0]
    balls_played = len(group)
    wickets = group['wickets'].max()
    
    # Team loses if:
    # 1. All out (10 wickets) before reaching target
    # 2. Didn't reach target in 120 balls (20 overs)
    # 3. Score less than target after all balls played
    if wickets == 10 or balls_played >= 120 or final_score < target:
        return 0
    # Team wins if they reach or exceed the target
    elif final_score >= target:
        return 1
    # Match incomplete or invalid
    else:
        return -1

def preprocess_data(df):
    """Preprocess the data for model training."""
    print(f"\nInitial data shape: {df.shape}")
    
    # Create a copy of the dataframe to avoid warnings
    df = df.copy()
    
    # Convert overs to balls using vectorized operation
    df['balls'] = (df['overs'].astype(int) * 6 + 
                  (df['overs'] % 1 * 10).round().astype(int))
    
    # Calculate balls remaining
    df['balls_remaining'] = 120 - df['balls']
    
    # Calculate cumulative runs and wickets for each match
    df['cum_runs'] = df.groupby('mid')['runs'].cumsum()
    df['cum_wickets'] = df.groupby('mid')['wickets'].cumsum()
    
    # Calculate wickets remaining and runs remaining
    df['wickets_remaining'] = 10 - df['cum_wickets']
    df['runs_remaining'] = df['total'] - df['cum_runs']
    
    # Calculate run rates safely
    df['current_run_rate'] = np.where(
        df['balls'] > 0,
        df['cum_runs'] * 6 / df['balls'],
        0
    )
    
    df['required_run_rate'] = np.where(
        df['balls_remaining'] > 0,
        df['runs_remaining'] * 6 / df['balls_remaining'],
        np.inf
    )
    
    # Calculate match results for each ball
    df['match_result'] = np.where(
        df['cum_runs'] >= df['total'], 1,  # Win if target reached
        np.where(
            (df['cum_wickets'] >= 10) |  # Loss if all out
            (df['balls'] >= 120) |       # Loss if overs completed
            ((df['total'] - df['cum_runs']) > (df['balls_remaining'] * 36/6)),  # Loss if required RR > 36
            0,  # Loss
            -1  # In progress
        )
    )
    
    # Keep only the rows where match result is determined
    df = df[df['match_result'] != -1].copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['bat_team', 'bowl_team', 'venue']
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Select features and target
    features = [
        'mid', 'bat_team', 'bowl_team', 'venue', 'total',
        'cum_runs', 'balls', 'cum_wickets', 'balls_remaining',
        'wickets_remaining', 'runs_remaining', 'current_run_rate',
        'required_run_rate'
    ]
    
    # Prepare final dataset
    X = df[features]
    y = df['match_result']
    
    print(f"\nFinal dataset shape - X: {X.shape}, y: {y.shape}")
    print(f"Number of unique matches: {len(df['mid'].unique())}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y, label_encoders

def save_preprocessed_data(X, y, output_path):
    """Save preprocessed data to CSV file."""
    # Combine features and target
    df = pd.concat([X, pd.Series(y, name='target')], axis=1)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total matches: {len(df['mid'].unique())}")
    print(f"Total rows: {len(df)}")
    print(f"Win rate by balls: {df.groupby('balls')['target'].mean().round(3)}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess the existing data
    df = load_data('data/ipl_matches.csv')
    X, y, label_encoders = preprocess_data(df)
    save_preprocessed_data(X, y, 'data/preprocessed_data.csv')
    
    # Save label encoders
    import joblib
    joblib.dump(label_encoders, 'models/label_encoders.joblib') 