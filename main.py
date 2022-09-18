import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

from scipy.spatial import distance

app = Flask(__name__, template_folder='Templates')

Scaler = pickle.load(open('norm_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction',methods=['POST'])
def getprediction():    

    input = [float(x) for x in request.form.values()]
    final_input = [np.array(input)]
    Scaled_data = Scaler.transform(final_input)
    
    with open('freezed_centroids.pkl', 'rb') as file:
        freezed_centroids = pickle.load(file)

    assigned_cluster = []
    l = []

    for i, this_segment in enumerate(freezed_centroids):
        dist = distance.euclidean(*Scaled_data, this_segment)
        l.append(dist)
        index_min = np.argmin(l)
        assigned_cluster.append(index_min)


    return render_template(
        'predict.html', result_value= f'Cluster = #{index_min}'
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
   
    