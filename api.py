# API to load clients' data, predict the scoring, product shap values

# Import required librairies
import flask
from flask import jsonify
import pickle
import numpy as np
import pandas as pd
import shap
import os

# Initializing the API
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Get the path where the data and model files are stored
path = os.path.dirname(os.path.realpath(__file__))

# Home page
@app.route('/', methods=['GET'])
def home():
    return """
<h1>OC - P7 - API</h1>
<p>API permettant de :
 - charger les données clients
 - charger le modèle de prédiction
 - prédire le scoring d'un client
 - produire les shap values pour expliquer la prédiction
 </p>
"""

# getting our trained model from a file we created earlier
model = pickle.load(open(path + "/model.pkl","rb"))
# getting the data
X = pd.read_csv(path + "/data_sample.csv")

# defining a route to get all clients data
@app.route("/data", methods=["GET"])
def get_data():
    df_all = X.to_dict("list")
    return jsonify(df_all)

# defining a route to get clients data and prediction
@app.route("/data/client/<client_id>", methods=["GET"])
def client_data(client_id):
    # filter the data thanks to the id from the request
    df_sample = X[X["SK_ID_CURR"] == int(client_id)]
    feature_array = np.asarray(df_sample.iloc[0,1:21])
    # calculate prediction and probability for this client
    df_sample["prediction"] = model.predict([feature_array]).tolist()[0]
    df_sample['proba_1'] = model.predict_proba([feature_array])[:,1].tolist()[0]
    # calculate features importance in this prediction
    explainer = shap.KernelExplainer(model.predict_proba, X.iloc[:,1:21]) 
    shap_values = explainer.shap_values(feature_array, l1_reg="aic")
    # add the shap values in the dataframe
    df_sample["expected"] = explainer.expected_value[1]
    new_line = [99999] + list(shap_values[1]) + [0,0,explainer.expected_value[1]]
    df_sample.loc[1] = new_line
    # create the dictionary to be sent
    sample = df_sample.to_dict("list")
    #returning sample and prediction objects as json
    return jsonify(sample)

# To remove when online :
if __name__ == '__main__':
    app.run(host='0.0.0.0')