import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="IPL Match Win Predictor",
    page_icon="🏏",
    layout="wide"
)

# Load the model and scaler
@st.cache_data
def load_model():
    model = joblib.load('models/ipl_predictor.joblib')
    scaler = joblib.load('models/scaler.joblib')
    label_encoders = joblib.load('models/label_encoders.joblib')
    return model, scaler, label_encoders

def predict_win_probability(model, scaler, label_encoders, input_data):
    """Predict win probability for given match situation."""
    # Create feature vector
    features = {
        'bat_team': label_encoders['bat_team'].transform([input_data['batting_team']])[0],
        'bowl_team': label_encoders['bowl_team'].transform([input_data['bowling_team']])[0],
        'venue': label_encoders['venue'].transform([input_data['venue']])[0],
        'total': float(input_data['target']),
        'cum_runs': float(input_data['current_score']),
        'balls': float(input_data['balls']),
        'cum_wickets': float(input_data['wickets']),
        'balls_remaining': 120 - float(input_data['balls']),
        'wickets_remaining': 10 - float(input_data['wickets']),
        'runs_remaining': float(input_data['target']) - float(input_data['current_score']),
    }
    
    # Calculate run rates
    features['current_run_rate'] = (features['cum_runs'] * 6 / features['balls']) if features['balls'] > 0 else 0
    features['required_run_rate'] = (features['runs_remaining'] * 6 / features['balls_remaining']) if features['balls_remaining'] > 0 else np.inf
    
    # Add match ID (dummy value for prediction)
    features['mid'] = 1
    
    # Create DataFrame with features in correct order
    feature_order = [
        'mid', 'bat_team', 'bowl_team', 'venue', 'total',
        'cum_runs', 'balls', 'cum_wickets', 'balls_remaining',
        'wickets_remaining', 'runs_remaining', 'current_run_rate',
        'required_run_rate'
    ]
    X = pd.DataFrame([features])[feature_order]
    
    # Scale features and make prediction
    X_scaled = scaler.transform(X)
    win_prob = model.predict_proba(X_scaled)[0][1]
    
    # Calculate additional stats
    overs = f"{input_data['balls'] // 6}.{input_data['balls'] % 6}"
    overs_remaining = f"{features['balls_remaining'] // 6}.{features['balls_remaining'] % 6}"
    
    return {
        'win_probability': win_prob,
        'current_run_rate': features['current_run_rate'],
        'required_run_rate': features['required_run_rate'],
        'overs': overs,
        'overs_remaining': overs_remaining
    }

def main():
    # Load model and encoders
    model, scaler, label_encoders = load_model()
    
    # Header
    st.title("🏏 IPL Match Win Predictor")
    st.markdown("""
    Predict the probability of a team winning an IPL match based on the current match situation.
    This predictor works best for second innings predictions.
    """)
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Match Details")
        
        # Get teams and venues from label encoders
        teams = sorted(label_encoders['bat_team'].classes_)
        venues = sorted(label_encoders['venue'].classes_)
        
        # Create input form
        batting_team = st.selectbox(
            "Batting Team",
            teams,
            index=0
        )
        
        bowling_team = st.selectbox(
            "Bowling Team",
            teams,
            index=1
        )
        
        venue = st.selectbox(
            "Venue",
            venues,
            index=0
        )
        
        target = st.number_input(
            "Target Score",
            min_value=0,
            max_value=300,
            value=180,
            step=1
        )
    
    with col2:
        st.subheader("Current Situation")
        
        current_score = st.number_input(
            "Current Score",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )
        
        # Input for overs and balls
        col_overs, col_balls = st.columns(2)
        with col_overs:
            overs = st.number_input(
                "Overs",
                min_value=0,
                max_value=19,
                value=10,
                step=1
            )
        with col_balls:
            balls = st.number_input(
                "Balls in Current Over",
                min_value=0,
                max_value=5,
                value=0,
                step=1
            )
        
        total_balls = overs * 6 + balls
        
        wickets = st.number_input(
            "Wickets Fallen",
            min_value=0,
            max_value=10,
            value=4,
            step=1
        )
    
    # Calculate button
    if st.button("Calculate Win Probability", type="primary"):
        # Create input data
        input_data = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'target': target,
            'current_score': current_score,
            'balls': total_balls,
            'wickets': wickets
        }
        
        # Make prediction
        result = predict_win_probability(model, scaler, label_encoders, input_data)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Batting Team Win Probability",
                f"{result['win_probability']*100:.1f}%",
                f"{(result['win_probability']*100 - 50):.1f}%"
            )
        
        with col2:
            st.metric(
                "Current Run Rate",
                f"{result['current_run_rate']:.2f}",
                f"vs Required: {result['required_run_rate']:.2f}"
            )
        
        with col3:
            st.metric(
                "Overs",
                f"{result['overs']}",
                f"Remaining: {result['overs_remaining']}"
            )
        
        # Create progress bars
        st.progress(result['win_probability'])
        st.progress(1 - result['win_probability'])
        
        # Display match situation
        st.markdown("### Match Situation")
        st.markdown(f"""
        - **Batting Team**: {batting_team}
        - **Bowling Team**: {bowling_team}
        - **Venue**: {venue}
        - **Target**: {target}
        - **Current Score**: {current_score}/{wickets}
        - **Overs**: {result['overs']} (Remaining: {result['overs_remaining']})
        - **Required Run Rate**: {result['required_run_rate']:.2f}
        """)

if __name__ == "__main__":
    main()