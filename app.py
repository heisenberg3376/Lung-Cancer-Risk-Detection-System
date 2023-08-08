from flask import Flask, render_template, request, redirect
import numpy as np
import pickle

xgb = pickle.load(open(r'C:\Users\Katta\OneDrive\Desktop\Flask Tutorial\XGB.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    #return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    ar = np.array(features,dtype='int64')
    pred = xgb.predict(ar.reshape(1,-1))
    val = '-'
    #print(ar)
    if(pred[0]==1):
        val = 'Very Low Risk of Lung Cancer'
    elif(pred[0]==2):
        val = 'Moderate Risk of Lung Cancer'
    elif(pred[0]==0):
        val = 'High Risk of Lung Cancer'
        
    return render_template('index.html',prediction_text=f'The Patient has {val}')

@app.route('/info.html')
def info():
    return render_template('info.html')





if __name__ == '__main__':
    app.run(debug=True)


