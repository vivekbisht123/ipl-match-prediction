# IPL Match Win Predictor

A machine learning-based web application that predicts the win probability of teams during IPL matches in the second innings.

## Features

- Real-time win probability prediction during second innings
- Interactive web interface using Streamlit
- Considers multiple match parameters:
  - Current score
  - Overs completed
  - Wickets fallen
  - Target score
  - Venue
  - Teams playing

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL

## Project Structure

- `data/`: Contains the IPL match dataset
- `models/`: Contains trained model files
- `preprocessing.py`: Data preprocessing and feature engineering
- `train_model.py`: Model training and evaluation
- `app.py`: Streamlit web application
- `requirements.txt`: Project dependencies

## Model Details

The prediction model uses Logistic Regression to estimate win probabilities based on:
- Current match situation
- Historical performance data
- Venue-specific factors
- Team-specific factors

## Contributing

Feel free to submit issues and enhancement requests!