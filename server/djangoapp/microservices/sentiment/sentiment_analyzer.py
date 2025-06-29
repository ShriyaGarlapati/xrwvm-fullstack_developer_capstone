from flask import Flask, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

@app.route('/analyze/<text>', methods=['GET'])
def analyze(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return jsonify(sentiment=sentiment)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
