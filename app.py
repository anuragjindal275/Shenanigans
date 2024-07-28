from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def load_and_preprocess_data():
    data = pd.read_csv('kerala.csv')
    le = LabelEncoder()
    data['FLOODS'] = le.fit_transform(data.FLOODS)
    data_inputs = data.iloc[:, 2:14].values
    data_outputs = data.iloc[:, -1].values
    return data_inputs, data_outputs

def train_models(data_inputs, data_outputs):
    x_train, x_test, y_train, y_test = train_test_split(data_inputs, data_outputs, test_size=0.3, random_state=0)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    minmax = MinMaxScaler()
    x_train_normal = minmax.fit_transform(x_train)
    x_test_normal = minmax.transform(x_test)

    Sc = StandardScaler()
    x_train_std = Sc.fit_transform(x_train)
    x_test_std = Sc.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train_normal, y_train)

    lr = LogisticRegression()
    lr.fit(x_train_std, y_train)

    svc = SVC(kernel='rbf', probability=True)
    svc.fit(x_train_normal, y_train)

    return knn, lr, svc, minmax, Sc

def predict_flood(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])
    probabilities = model.predict_proba(input_data_scaled)
    return probabilities[0][1] * 100  # Return the probability of class 1 (flood)

data_inputs, data_outputs = load_and_preprocess_data()
knn, lr, svc, minmax, Sc = train_models(data_inputs, data_outputs)

@app.route('/', methods=['GET', 'POST'])
def index():
    knn_pred = None
    lr_pred = None
    svc_pred = None
    graph_url = None
    if request.method == 'POST':
        input_data = [float(x) for x in request.form.values()]
        knn_pred = predict_flood(knn, minmax, input_data)
        lr_pred = predict_flood(lr, Sc, input_data)
        svc_pred = predict_flood(svc, minmax, input_data)
        
        # Create a bar chart of the predictions
        fig, ax = plt.subplots()
        labels = ['KNN', 'Logistic Regression', 'SVM']
        percentages = [knn_pred, lr_pred, svc_pred]
        ax.bar(labels, percentages, color=['blue', 'green', 'red'])
        ax.set_ylabel('Flood Prediction Percentage')
        ax.set_title('Flood Prediction by Different Models')
        plt.ylim(0, 100)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('index.html', knn_pred=knn_pred, lr_pred=lr_pred, svc_pred=svc_pred, graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)
