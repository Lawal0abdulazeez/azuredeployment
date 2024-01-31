import joblib  # To read our pickled model and scaler
from flask import Flask, request, jsonify, url_for, render_template  # Importing falsk framework and neccesary libraries
import numpy as np #Importing numpy for data conversion and reshaping

app=Flask(__name__, static_folder='/static')  #initializing flask

## Load the model
regmodel= joblib.load(open('regmodel.h5', 'rb'))  # Loading our model from the pickled model file and can now be used in our web app
scalar=joblib.load(open('scaling.h5','rb'))      # Loading the scaler from the pickled scaling file and can now be used in our web app

@app.route('/')  # our home page
def home():
    return render_template('home.html') # Returns an html file that collect the input of our data and have a button for prediction


@app.route('/predict_api', methods=['POST'])   # An local api that can be used with postman 
def predict_api():
    data=request.json['data']  # request a json file from the host(Postman)
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) 
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1)) # reshaping the size to fit into the model
    output=regmodel.predict(new_data)  # predicting and saving our predicted values in a variable
    print(output)
    return jsonify(output[0])   # return our predicted value as json to be shown in Postman

@app.route('/predict', methods=['POST'])  # A web page links to our form and prediction button in our nhtml file
def predict():
    data=[float(x) for x in request.form.values()]  # Turning our form values to float and saving the in a variable
    final_input=scalar.transform(np.array(data).reshape(1,-1))  # reshaping the size to fit into the model
    print(final_input)
    output=regmodel.predict(final_input)[0]     # predicting and saving our predicted values in a variable
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))     # return our predicted value and the basic html page




if __name__=="__main__":   # To start the app
    app.run(debug=True)    # Enabling debugging