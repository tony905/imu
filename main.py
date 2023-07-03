from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def Home():
    return "Hello World"

@app.route('/predict',methods=['POST'])

def predict():
    AccX = float(request.form.get('AccX'))
    AccY = float(request.form.get('AccY'))
    AccZ = float(request.form.get('AccZ'))
    GyrX = float(request.form.get('GyrX'))
    GyrY = float(request.form.get('GyrY'))
    GyrZ = float(request.form.get('GyrZ'))

    input_query = np.array([[AccX, AccY, AccZ, GyrX, GyrY, GyrZ]])

    result = model.predict(input_query)[0]

    return jsonify({'Fall': str(result)})



if __name__ == '__main__':
    app.run(debug=True)
