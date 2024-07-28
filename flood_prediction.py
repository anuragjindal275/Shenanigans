import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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
    prediction = model.predict(input_data_scaled)
    return prediction

if __name__ == "__main__":
    data_inputs, data_outputs = load_and_preprocess_data()
    knn, lr, svc, minmax, Sc = train_models(data_inputs, data_outputs)

    example_input = [28.7, 44.7, 51.6, 160.0, 174.7, 824.6, 743.0, 357.5, 197.7, 266.9, 350.8, 48.4]

    knn_prediction = predict_flood(knn, minmax, example_input)
    lr_prediction = predict_flood(lr, Sc, example_input)
    svc_prediction = predict_flood(svc, minmax, example_input)

    print("KNN Prediction:", knn_prediction)
    print("Logistic Regression Prediction:", lr_prediction)
    print("SVM Prediction:", svc_prediction)
