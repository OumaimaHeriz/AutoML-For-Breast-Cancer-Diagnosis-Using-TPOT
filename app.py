import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))  # Adjust the path if necessary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean'])
        ]
        input_data = pd.DataFrame([data], columns=[
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])
        output = model.predict(input_data)[0]
        result = 'Malignant' if output == 1 else 'Benign'
        return render_template("result.html", result=result)
    except KeyError as e:
        return f"Missing or invalid input: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
