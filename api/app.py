import sys
import traceback
from flask import Flask, request, jsonify
import joblib
import pandas as pd  

app = Flask(__name__)

try:
    glm = joblib.load('model/glm.joblib') # load model
    print('Model loaded successfully')
except Exception as e:
    print('Error loading model:', e)
    glm = None

@app.route('/predict', methods=['POST']) 
def predict():
    if glm:
        try:
            json_data = request.get_json()
            
            if not json_data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            n = 1 # set n = 1 for the case of a single prediction
            
            if isinstance(json_data, list): # determine if it is a batch request
                n = len(json_data)

            df = pd.DataFrame(json_data, index=[list(range(n))]) 
            prediction = pd.DataFrame(glm.predict(df.astype(float))).rename(columns={0:'phat'})

            prediction['business_outcome'] = prediction.apply(lambda row : 1 if row.phat > 0.71 and row.phat <= 0.995 else 0, axis=1)
            
            model_output = pd.concat([prediction, df], axis=1).to_dict(orient='records')

            return jsonify(model_output)
        
        except Exception as e:
            return jsonify({'exception': str(e), 'trace': traceback.format_exc()}), 500
    else:
        print ('Model did not load successfully')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # command line arg
    except:
        port = 1313 # if no port is provided, set to this
    app.run(port=port, debug=True)