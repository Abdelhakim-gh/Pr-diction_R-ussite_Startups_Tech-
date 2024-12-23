import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

hr_data = pd.read_csv('HR_comma_sep.csv')
hr_data.head()
hr_data.describe()
hr_data.info()
#il n'y a pas de données manquantes. 

#remplacer le nom de l attribut sales par departement
hr_data.rename(columns={'sales':'department'}, inplace=True)

hr_data['department'].unique()
hr_data['salary'].unique()
hr_data.columns
#gerer les dummy variables
hr_data_new = pd.get_dummies(hr_data, ['department', 'salary'] ,drop_first = True)
hr_data_new.columns
hr_data_new.head()
hr_data['department'].unique()

# matrix de Correlation 
sns.heatmap(hr_data.corr(), annot=True)

from sklearn.model_selection import train_test_split
X = hr_data_new.drop('left', axis=1)
y = hr_data_new['left']
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
#regardons le score sur nos données d'entraînement 
dt.score(X_train, y_train)
# = 1 c est parfait, mais verifions que notre modele est stable avec la validation croisée
from sklearn.model_selection import cross_val_score

cross_val_score(dt, X_train, y_train, cv=10)

cross_val_score(dt,X_train,y_train,cv=10).mean()

#Faisons des prédictions et vérifions les performances du modèle sur le testing set
y_pred = dt.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve, f1_score,roc_auc_score

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test,y_pred)

precision_score(y_test,y_pred)

recall_score(y_test,y_pred)

f1_score(y_test,y_pred)

fpr,tpr,thresholds =  roc_curve(y_test,y_pred)
plt.plot(fpr,tpr,"b")
plt.plot([0,1],[0,1],"r-")
roc_auc_score(y_test,y_pred)

#maintenant examinons les attributs les plus importantes, ou ceuxs qui ont le plus 
#d'influence pour déterminer si un employé quitte (ou reste) dans l entreprise. 

importances = dt.feature_importances_
print("Feature importances: \n")
for f in range(len(X.columns)):
    print('•', X.columns[f], ":", importances[f])
    

from sklearn.ensemble import RandomForestClassifier
modelRFC = RandomForestClassifier(n_estimators = 1000,max_depth = 6, min_samples_split = 15)

modelRFC.fit(X,y)
modelRFC.score(X,y)*100

modelRFC.fit(X_train,y_train)
modelRFC.score(X_train,y_train)*100

y_pred = modelRFC.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

precision_score(y_test,y_pred)

recall_score(y_test,y_pred)

f1_score(y_test,y_pred)

cm = confusion_matrix(y_test, y_pred)


n_estimators = [int(x) for x in np.linspace(start = 170, stop = 175)]
max_depth = [2,20]
min_samples_split = [2, 10]

param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               }

rf_Model = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid)
rf_Grid.fit(X_train, y_train)
rf_Grid.best_params_
#{'n_estimators': 171}
print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')
#Train Accuracy - : 1.000
#Test Accuracy - : 0.990

# TP 1 : A refaire le RandomForestClassifier avec RandomizedSearchCV et comparer les resultats

#pip install xgboost dans anaconda prompt
from xgboost import XGBClassifier
modelXGB = XGBClassifier(max_depth = 5,learning_rate = 0.01)

modelXGB.fit(X,y)

modelXGB.score(X,y)
y2_pred = modelXGB.predict(X_test)
accuracy_score(y_test,y2_pred)






    