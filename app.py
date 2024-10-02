from flask import Flask, request, jsonify, render_template
import pickle
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

app = Flask(__name__)

# Load the scaler and the XGBoost model trained on top features
with open('scaler_top.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('xgboost_model_top.pkl', 'rb') as f:
    model = pickle.load(f)

# Define top features (ensure this matches the top_features used during training)
top_features = ['PPE', 'spread1', 'MDVP:Fo(Hz)', 'MDVP:APQ', 'MDVP:Flo(Hz)', 
                'Shimmer:APQ5', 'Shimmer:APQ3', 'MDVP:Shimmer(dB)', 'MDVP:Shimmer', 'NHR']

# Define label mapping
label_mapping = {0: 'Healthy', 1: 'Parkinson\'s Disease'}

@app.route('/')
def home():
    return render_template('home.html', top_features=top_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract feature values from the form
        feature_values = []
        for feature in top_features:
            # Get the value from the form; handle missing or invalid inputs
            value = request.form.get(feature)
            if value is None or value.strip() == '':
                raise ValueError(f"Missing value for {feature}")
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Invalid value for {feature}: '{value}' is not a number.")
            feature_values.append(value)
        
        print("Collected Feature Values:", feature_values)  # Debugging
        
        # Convert to numpy array and reshape
        input_data = np.array(feature_values).reshape(1, -1)
        print("Input Data Shape:", input_data.shape)  # Debugging
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        print("Scaled Input Data Shape:", input_scaled.shape)  # Debugging
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        
        # Map the prediction to label
        predicted_label = label_mapping.get(prediction, "Unknown")


        # Generate a probability bar graph
        fig, ax = plt.subplots()
        categories = ['Healthy', 'Parkinson\'s Disease']
        probabilities = [1 - prediction_proba, prediction_proba]
        colors = ['green', 'red'] if predicted_label == 'Parkinson\'s Disease' else ['green', 'red']
        
        ax.bar(categories, probabilities, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probability')
        
        # Convert plot to PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)  # Close the figure to free memory
        



        # Prepare the result
        result = {
            'prediction': predicted_label,
            'probability': round(prediction_proba * 100, 2),
            'plot_url': plot_url
        }
        
        return render_template('result.html', result=result)
    
    except ValueError as ve:
        # Handle invalid input values
        error_message = f"Invalid input: {ve}"
        return render_template('error.html', error=error_message)
    
    except Exception as e:
        # Handle other exceptions
        error_message = "An error occurred during prediction. Please check your inputs and try again."
        return render_template('error.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
