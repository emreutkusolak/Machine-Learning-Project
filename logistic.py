from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sn

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
#plt.show()

feature_cols = ['blueKills','redKills']
target_col = 'blueWins'
dataFrame = pd.DataFrame(dataset, columns=col_names)

X = dataFrame[feature_cols] # Independent Variables
y = dataFrame[target_col] # Dependent variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()

