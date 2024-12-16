from flask import Flask, request, jsonify, render_template
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and vectorizer
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')  # Renders an HTML form for input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input message from the request
        input_sms = request.form['message']

        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Response
        response = "Spam" if result == 1 else "Not Spam"

        return jsonify({"prediction": response})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
