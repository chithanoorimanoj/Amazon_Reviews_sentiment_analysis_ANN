# Amazon Review Sentiment Analysis Using ANN 

## Project Overview
Amazon Review Sentiment Analysis is a Machine Learning project that analyzes customer reviews and predicts their sentiment. It uses Natural Language Processing (NLP) techniques to preprocess text and classify reviews into sentiment categories.

The project is deployed using Streamlit, providing a simple web interface where users can enter a review and receive a sentiment prediction.

## Features
- Predict sentiment from customer reviews.
- Text preprocessing using NLP techniques.
- Tokenization, stopword removal, and lemmatization.
- TF-IDF feature extraction (if used in the trained model).
- Machine Learning-based sentiment classification.
- Interactive web application built with Streamlit.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit

## Model
The project uses a trained Machine Learning model saved as `model.joblib`. The application loads this model to generate predictions from user-provided reviews.

## Deployment
The application can be deployed using Streamlit Community Cloud by connecting this GitHub repository and selecting `app.py` as the main application file.

## Applications
- Customer feedback analysis
- Product review analysis
- Understanding customer opinions
- Supporting ecommerce businesses in analyzing customer sentiment

## Future Improvements
- Improve model performance through hyperparameter tuning.
- Evaluate the model using precision, recall, F1-score, and accuracy.
- Add visualizations of customer sentiment.
- Extend the application to analyze multiple reviews.

## Author
Manoj
