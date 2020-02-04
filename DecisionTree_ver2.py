from sklearn import linear_model, metrics, preprocessing, model_selection
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from dtreeviz.trees import dtreeviz
import unittest

#データセットを読み込み,四分位数からラベルを決定
def quartile(df):
    #df = pd.read_csv('All.csv')
    q0 = df["Happiness Score"].quantile(0.00)
    q1 = df["Happiness Score"].quantile(0.25)
    q2 = df["Happiness Score"].quantile(0.50)
    q3 = df["Happiness Score"].quantile(0.75)

    #四分位数からラベルを決定
    label = []
    qwe = df.iloc[:, :]
    for i in range(len(qwe)):
        if df.iloc[i]["Happiness Score"] >= q0 and df.iloc[i]["Happiness Score"] < q1:
            label.append(0)
        elif df.iloc[i]["Happiness Score"] >= q1 and df.iloc[i]["Happiness Score"] < q2:
            label.append(1)
        elif df.iloc[i]["Happiness Score"] >= q2 and df.iloc[i]["Happiness Score"] < q3:
            label.append(2)
        else:
            label.append(3)
    s = pd.DataFrame(label)
    df["label"]=s
    
    return df

class Testquartile(unittest.TestCase):

    def test_quartile(self):
        df = pd.DataFrame({"Happiness Score" : np.array([0,1,2,3])})
        exp = df
        exp["label"] = pd.DataFrame([0,1,2,3])
        act = quartile(df)
        assert_frame_equal(exp, act)

if __name__ == "__main__":
    unittest.main()
