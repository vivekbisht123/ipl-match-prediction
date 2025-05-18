import pandas as pd
import numpy as np
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download IPL match data from Kaggle."""
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset
    dataset = 'nowke9/ipldata'
    api.dataset_download_files(dataset, path='data', unzip=True)
    
    print("Dataset downloaded successfully!")

def prepare_dataset():
    """Prepare the dataset for our model."""
    # Read the matches data
    matches_df = pd.read_csv('data/matches.csv')
    deliveries_df = pd.read_csv('data/deliveries.csv')
    
    # Merge matches and deliveries data
    merged_df = pd.merge(deliveries_df, matches_df[['id', 'season', 'city', 'venue']], 
                        left_on='match_id', right_on='id', how='left')
    
    # Calculate match situation features
    match_situations = []
    
    for match_id in merged_df['match_id'].unique():
        match_data = merged_df[merged_df['match_id'] == match_id]
        
        # Get match details
        match_info = matches_df[matches_df['id'] == match_id].iloc[0]
        target = match_info['total_runs'] if match_info['inning'] == 2 else None
        
        if target is None:
            continue
            
        # Calculate cumulative runs and wickets for each over
        for inning in [1, 2]:
            inning_data = match_data[match_data['inning'] == inning]
            if inning_data.empty:
                continue
                
            runs = 0
            wickets = 0
            current_over = 0
            
            for _, ball in inning_data.iterrows():
                if ball['over'] > current_over:
                    # Save the situation at the end of each over
                    match_situations.append({
                        'match_id': match_id,
                        'inning': inning,
                        'batting_team': ball['batting_team'],
                        'bowling_team': ball['bowling_team'],
                        'city': match_info['city'],
                        'venue': match_info['venue'],
                        'target': target if inning == 2 else None,
                        'total_runs': runs,
                        'over': ball['over'],
                        'wicket': wickets,
                        'result': 1 if match_info['winner'] == ball['batting_team'] else 0
                    })
                    current_over = ball['over']
                
                runs += ball['total_runs']
                if ball['player_dismissed'] is not None and not pd.isna(ball['player_dismissed']):
                    wickets += 1
    
    # Convert to DataFrame
    final_df = pd.DataFrame(match_situations)
    
    # Save the prepared dataset
    final_df.to_csv('data/ipl_matches_processed.csv', index=False)
    print("Dataset prepared and saved successfully!")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download and prepare the dataset
    download_dataset()
    prepare_dataset()