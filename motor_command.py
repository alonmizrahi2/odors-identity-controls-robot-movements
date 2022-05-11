
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from Motor import *


def Motor_Command(Classifier_output):

    if Classifier_output == 0:  # ('Ger')
        PWM.setMotorModel(1000, 1000, 1000, 1000)  # Forward
        print("The car is moving forward")
        time.sleep(1)
    elif Classifier_output == -1:  # ('Rose')
        PWM.setMotorModel(-1000, -1000, 1500, 1500)  # Left
        print("The car is turning left")
        time.sleep(1)
    elif Classifier_output == 1:  # ('Benz')
        PWM.setMotorModel(1500, 1500, -1000, -1000)  # Right
        print("The car is turning right")
        time.sleep(1)
    else:
        print('No match')

    PWM.setMotorModel(0, 0, 0, 0)  # Stop
    print("\nEnd session")

if __name__ == '__main__':

    test = pd.read_excel('../data/Test_odors_and_control.xlsx')
    data_test = test.iloc[:, :-4]
    labels_test = test['label']

    for index, row in data_test.iterrows():
        scaler = MinMaxScaler()
        scaler.fit(data_test.iloc[index,:].values.reshape(-1, 1))
        data_test.iloc[index,:] = scaler.transform(data_test.iloc[index,:].values.reshape(-1, 1)).reshape(1,-1)

    lab_ = []
    for elem in labels_test:
        if elem == '4-Rose':
            lab_.append(-1)
        elif elem == '1-Benz':
            lab_.append(1)
        elif elem == '6-Ger':
            lab_.append(0)
        else:
            lab_.append(None)

    labels_test = pd.DataFrame(lab_[:-3], columns=['Label'])

    y_test = labels_test.Label.values
    # X_test = data_test.values[:-3, :]
    X_test = data_test.values

    with open('../data/svc_model.pkl', 'rb') as h:
        svc = pickle.load(h)

    pred = svc.predict(X_test)

    PWM=Motor()

    for prediction in pred:
        Motor_Command(prediction)
    #PWM.setMotorModel(1500, 1500, -1500, -1500)
    #time.sleep(3)
    #PWM.setMotorModel(0, 0, 0, 0)
