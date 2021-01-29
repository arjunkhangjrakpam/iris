import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
#os.chdir("/home/fyiacademy/ARJUN/SELF LEARNING/ML/VARIOUS_PYTHON_TASKS/ENTIRE ML PIPELINE with Loan prediction example/CLASSIFICATION")
os.chdir(os.getcwd())

app = Flask(__name__)
model = pickle.load(open('iris_data_classification.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['sepal length (cm)',
                     'sepal width (cm)',
                     'petal length (cm)',
                     'petal width (cm)']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "Setosa"
    elif output == 1: 
        res_val = "Versicolour"
    elif output == 2:
        res_val = "Virginica"
        

    return render_template('index.html', prediction_text=' The plant is Iris-{}'.format(res_val))

if __name__ == "__main__":
    app.run()
