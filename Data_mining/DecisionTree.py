from sklearn import linear_model, metrics, preprocessing, model_selection
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from dtreeviz.trees import dtreeviz

#データセットを読み込み,四分位数からラベルを決定
def quartile(df):
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

#地域から"region_label" を追加
def addResionLabel(df):
    region_label = []
    for i in range(157):
        if df.iloc[i]["Region"]=="Western Europe" or df.iloc[i]["Region"]=="Central and Eastern Europe":
            region_label.append(3)
        elif df.iloc[i]["Region"]=="North America":
            region_label.append(4)
        elif df.iloc[i]["Region"]=="Latin America and Caribbean":
            region_label.append(2)
        elif df.iloc[i]["Region"]=="Australia and New Zealand":
            region_label.append(5)
        elif df.iloc[i]["Region"]=="Sub-Saharan Africa":
            region_label.append(0)
        else:
            region_label.append(1)
    rl = pd.DataFrame(region_label)
    df["region label"]=rl
    
    return df

#機械学習で分類する
df = pd.read_csv("2016.csv")
df = addResionLabel(df)
df = quartile(df)

dummy_region = pd.get_dummies(df["region label"])
X1 = df.iloc[:, 4:15]
X = pd.concat([X1, dummy_region],axis=1)
y = df["label"]
clf_result=tree.DecisionTreeClassifier(max_depth=3)

#トレーニングデータとテストデータに分けて実行する
X_train, X_test, y_train, y_test=model_selection.train_test_split(X, y, test_size=0.5)
clf_result.fit(X_train, y_train)

#正答率を求める
pre1=clf_result.predict(X_train)
ac_score1 = metrics.accuracy_score(y_train, pre1)
print("トレーニングデータ正答率 = ", ac_score1)
pre2=clf_result.predict(X_test)
ac_score2 = metrics.accuracy_score(y_test, pre2)
print("テストデータ正答率 = ", ac_score2)

"""
#重要度の可視化
features = []
for s in range(17):
    features.append(X.columns[s])
    
n_features = X.shape[1]
plt.barh(range(n_features), clf_result.feature_importances_, align="center")
plt.yticks(np.arange(n_features),features)
plt.xlabel("importance")
plt.ylabel("Feature value")
plt.savefig("barh.png")

#決定木の可視化
viz = dtreeviz(
    clf_result,
    X.values,
    y.values,
    target_name = "variety",
    feature_names = features,
    class_names = ["0","1","2","3"])
    
viz.view()
"""
