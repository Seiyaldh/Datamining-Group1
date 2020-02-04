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
    for i in range(len(df)):
        if df.iloc[i]["Region"]=="Western Europe" or df.iloc[i]["Region"]=="Central and Eastern Europe":
            region_label.append("Europe")
        elif df.iloc[i]["Region"]=="North America":
            region_label.append("North America")
        elif df.iloc[i]["Region"]=="Latin America and Caribbean":
            region_label.append("South America")
        elif df.iloc[i]["Region"]=="Australia and New Zealand":
            region_label.append("Oceania")
        elif df.iloc[i]["Region"]=="Sub-Saharan Africa":
            region_label.append("Africa")
        else:
            region_label.append("Asia")
    rl = pd.DataFrame(region_label)
    df["region label"]=rl
    
    return df

#機械学習で分類する
df = pd.read_csv("All.csv")
df = addResionLabel(df)
df = quartile(df)

df2 = pd.read_csv("2019.csv")
df2 = addResionLabel(df2)
df2 = quartile(df2)
dummy_region = pd.get_dummies(df["region label"])
dummy_region2 = pd.get_dummies(df2["region label"])

All_X1 = df.iloc[:, df.columns.get_loc("Economy (GDP per Capita)"):df.columns.get_loc("UnenploymentRate")+1]
All_X = pd.concat([All_X1, dummy_region],axis=1)
All_y = df["label"]

X1_2015 = df2.iloc[:, df2.columns.get_loc("Economy (GDP per Capita)"):df2.columns.get_loc("UnenploymentRate")+1]
X_2015 = pd.concat([X1_2015, dummy_region2],axis=1)
y_2015 = df2["label"]

clf_result=tree.DecisionTreeClassifier(max_depth=3)

#トレーニングデータとテストデータに分けて実行する
#X_train, X_test, y_train, y_test=model_selection.train_test_split(X, y, test_size=0.5)
clf_result.fit(All_X, All_y)

#正答率を求める
#pre1=clf_result.predict(All_X)
#ac_score1 = metrics.accuracy_score(y_train, pre1)
#print("トレーニングデータ正答率 = ", ac_score1)
pre=clf_result.predict(X_2015)
ac_score = metrics.accuracy_score(y_2015, pre)
print("正答率 = ", ac_score)
"""
#重要度の可視化
features = []
for s in range(len(X.columns)):
    features.append(X.columns[s])
    
n_features = X.shape[1]
plt.barh(range(n_features), clf_result.feature_importances_, align="center")
plt.yticks(np.arange(n_features),features)
plt.xlabel("importance")
plt.ylabel("Feature value")
plt.savefig("barh.png")
"""
#決定木の可視化
features = []
for s in range(len(X_2015.columns)):
    features.append(X_2015.columns[s])
    
viz = dtreeviz(
    clf_result,
    X_2015.values,
    y_2015.values,
    target_name = "variety",
    feature_names = features,
    class_names = ["0","1","2","3"])
    
viz.view()
