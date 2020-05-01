from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
col_names = ['gameId','blueWins','blueWardsPlaced',
             'blueWardsDestroyed','blueFirstBlood','blueKills',
             'blueDeaths','blueAssists','blueEliteMonsters',
             'blueDragons','blueHeralds','blueTowersDestroyed',
             'blueTotalGold','blueAvgLevel','blueTotalExperience',
             'blueTotalMinionsKilled','blueTotalJungleMinionsKilled',
             'blueGoldDiff','blueExperienceDiff','blueCSPerMin',
             'blueGoldPerMin','redWardsPlaced','redWardsDestroyed',
             'redFirstBlood','redKills','redDeaths','redAssists',
             'redEliteMonsters','redDragons','redHeralds',
             'redTowersDestroyed','redTotalGold','redAvgLevel',
             'redTotalExperience','redTotalMinionsKilled',
             'redTotalJungleMinionsKilled','redGoldDiff',
             'redExperienceDiff','redCSPerMin','redGoldPerMin']
dataset = pd.read_csv("high_diamond_ranked_10min.csv",sep=',', header=None, names=col_names)[1:]
dataset = dataset.apply(pd.to_numeric, errors='ignore')
plt.hist(dataset['blueWins'], bins=3)
#plt.show()

feature_cols = ['blueKills','redKills']
target_col = ['blueWins']

X = dataset[feature_cols] # Features
X=X.iloc[:,:2].values
y = dataset[target_col] # Target variable
y=y.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
error = []
# Calculating error for K values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
print(classification_report(y_test, pred_i))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
