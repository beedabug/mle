import sys
import traceback
from urllib import request
from flask import Flask, jsonify
import joblib
import pandas as pd 

from model import glm   

app = Flask(__name__)


@app.route('/predict', methods=['POST']) 
def predict():
    if glm:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(glm.predict(query))

            return jsonify({'prediction': prediction})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 1313 # if no port is provided, set to this
    glm = joblib.load('model/glm.joblib') # load model
    print ('Model loaded')
    model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)