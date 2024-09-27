from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    # Preprocess and vectorize the input message
    message_cleaned = message.lower().replace(r'[^\w\s]', '')
    message_vector = vectorizer.transform([message_cleaned])
    
    # Make the prediction
    prediction = model.predict(message_vector)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
