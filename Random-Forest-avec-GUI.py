import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import tkinter as tk 

#gmat : (Graduate Management Admission Test) qui varient de 540 à 790. Le GMAT est un test standardisé largement utilisé pour l'admission aux programmes MBA.
#gpa : La moyenne cumulative (Grade Point Average) sur une échelle de 4.0
#work_experience : Années d'expérience professionnelle, Varie de 1 à 6 ans
#age : L'âge des candidats Varie de 22 à 31 ans
#admitted : Le statut d'admission codé en 3 catégories 0 : Non admis 1 : Probablement liste d'attente ou admission conditionnelle 2 : Admis
candidates = {'gmat': [780,750,690,710,780,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,760,640,620,660,660,680,650,670,580,590,790],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'age': [25,28,24,27,26,31,24,25,28,23,25,27,30,28,26,23,29,31,26,26,25,24,28,23,25,29,28,26,30,30,23,24,27,29,28,22,23,24,28,31],
              'admitted': [2,2,1,2,2,2,0,2,2,0,0,2,2,1,2,0,0,1,0,0,1,0,0,0,0,1,1,0,1,2,0,0,1,1,1,0,0,0,0,2]
              }

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','age','admitted'])
print (df)

X = df[['gmat', 'gpa','work_experience','age']]
y = df['admitted']
y.unique()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

prediction = clf.predict([[730,3.7,4,27]]) 
print ('Predicted Result: ', prediction)

# tkinter GUI
root= tk.Tk()
root.title("Random forest avec GUI") 

canvas1 = tk.Canvas(root, width = 500, height = 350)
canvas1.pack()

# GMAT
label1 = tk.Label(root, text='            GMAT:')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root)
canvas1.create_window(270, 100, window=entry1)

# GPA
label2 = tk.Label(root, text='GPA:     ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root)
canvas1.create_window(270, 120, window=entry2)

# work_experience
label3 = tk.Label(root, text='     Work Experience: ')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry (root)
canvas1.create_window(270, 140, window=entry3)

# Age input
label4 = tk.Label(root, text='Age:                               ')
canvas1.create_window(160, 160, window=label4)

entry4 = tk.Entry (root)
canvas1.create_window(270, 160, window=entry4)

def values(): 
    global gmat
    gmat = float(entry1.get()) 
    
    global gpa
    gpa = float(entry2.get()) 
    
    global work_experience
    work_experience = float(entry3.get()) 
    
    global age
    age = float(entry4.get()) 
    
    Prediction_result  = ('  Predicted Result: ', clf.predict([[gmat,gpa,work_experience,age]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='sky blue')
    canvas1.create_window(270, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='      Predict      ',command=values, bg='green', fg='white', font=11)
canvas1.create_window(270, 220, window=button1)
 
root.mainloop()

