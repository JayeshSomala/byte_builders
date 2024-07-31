from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import string

app = Flask(__name__)
CORS(app)

# Load pre-trained model
with open('model_category_svm.pkl', 'rb') as f:
    model_category_svm = pickle.load(f)

# Define a function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inquiry = request.form['inquiry']
    processed_inquiry = preprocess_text(inquiry)
    
    # Predict category
    category = model_category_svm.predict([processed_inquiry])[0]
    
    return jsonify({
        'category': category
    })

if __name__ == '__main__':
    app.run(debug=True)
