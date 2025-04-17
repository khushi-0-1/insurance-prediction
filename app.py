from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

app = Flask(__name__)

# Load the dataset
insurance_dataset = pd.read_csv('insurance.csv')

# Encoding categorical columns
insurance_dataset = insurance_dataset.replace({'sex': {'male': 0, 'female': 1}})
insurance_dataset = insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}})
insurance_dataset = insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}})

# Splitting the data
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Prepare input data
        input_data = (age, sex, bmi, children, smoker, region)
        feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        input_data_as_dataframe = pd.DataFrame([input_data], columns=feature_names)

        # Prediction
        prediction = regressor.predict(input_data_as_dataframe)

        # Display result on result page
        return render_template('result.html', prediction=round(prediction[0], 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render or Railway will inject PORT
    app.run(host='0.0.0.0', port=port)

