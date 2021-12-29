from flask import Flask, request, render_template
import pickle
import boto3
import numpy as np

BUCKET_NAME = 'noshowml'
MODEL_FILE_NAME = 'model.pkl'

# load model
# model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)
S3 = boto3.client('s3', region_name='us-east-2')
headers =  {'x-api-key': 'c0p6JB8gCh6NJvqyGbTri33kAzULFoxkpc16Ix10'}

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

# routes
@app.route('/predict', methods=['GET','POST'])
def predict():
    Pclass = request.form['Pclass']
    Age = request.form['Age']
    SibSp = request.form['SibSp']
    Fare = request.form['Fare']
    data_df = [[Pclass, Age, SibSp, Fare]]
    result = predict(data_df)
    result = np.round(result)
    if result[0]==1:
        return "Survived"
    else:
        return "Not Survived"

def predict(data):
    response = S3.get_object(Bucket = BUCKET_NAME,Key = MODEL_FILE_NAME )
    model_str = response['Body'].read()
    model = pickle.loads(model_str)
    return model.predict(data)

if __name__ == '__main__':
    app.run(debug=True)
