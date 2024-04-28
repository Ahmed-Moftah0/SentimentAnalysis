from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

custom_classes = ["Love", "Joy", "Sadness","Anger","Surprise","Fear"]

# Load pipeline with sentiment analysis task
sentiment_analysis = pipeline("text-classification",model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

@app.route('/')
def home():
    return "Welcome to Sentiment Analysis API!"

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text')  # Use get() method to avoid KeyError
    if text:
        result = sentiment_analysis(text)
        highest_data = max(result[0], key=lambda x: x['score'])
        highest_label = highest_data['label']
        highest_score = highest_data['score']

        return jsonify({'sentiment': highest_label , 'Score':highest_score})
    else:
        return jsonify({'error': 'Text not provided'}), 400
if __name__ == '__main__':
    app.run(debug=True)
