
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint

class Data:
    def __init__(self) -> None:
        self.data = pd.read_csv('DataSets/diabetes_prediction_dataset.csv')


class IOdata(Data):
    def __init__(self) -> None:
        super().__init__()

    def get_inputs(self):
        return self.data.drop(columns=['blood_glucose_level', 'HbA1c_level', 'smoking_history', 'gender', 'diabetes'])
    
    def get_outputs(self):
        return self.data['diabetes']
    

class Model:
    def __init__(self) -> None:
        self.model = DecisionTreeClassifier()
            

if __name__ == '__main__':
    data = IOdata()
    X = data.get_inputs()
    y = data.get_outputs()

    try:
        m = Model().model
        m.fit(X, y)
    except:
        0
    
    finally:

        new_dates = [[
            78.0,   #age
            1,      #hypertension
            1,      #heart_disease
            27.45   #bmi
            ]]
        my_result = m.predict(new_dates)
        if my_result == [1]:
            pprint('HIGH level risk!')
        
        elif my_result == [0]:
            pprint('LOW level risk!')




    
    
