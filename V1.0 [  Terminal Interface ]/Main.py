import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

data = pd.read_csv('Training_Data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model = RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)



clearConsole()
age = float(input("Enter Age: "))
clearConsole()
gender = float(input("Gender:\n1.Male\n2.Female\nEnter Your Choice: "))
clearConsole()
cpt = float(input("Chest Pain Type:\n0.Typical Angina\n1.Atypical Angina\n2.Non-anginal Pain\n3.Asymptomatic\nEnter Your Choice: "))
clearConsole()
rbp = float(input("Resting BP(mm Hg): "))
clearConsole()
cotl = float(input("Cholestrol(mg/dl): "))
clearConsole()
fbs = float(input("Is Fasting Blood Sugar > 120 mg/dl(FBS):\n0.Yes\n1.No\nEnter Your Choice: "))
clearConsole()
recg = float(input("Resting ECG:\n0.Normal\n1.ST-T\n2.Probable\nEnter Your Choice: "))
clearConsole()
mhra = float(input("Max Heart Rate Achieved: "))
clearConsole()
eia = float(input("Exercise Indulge:\n0.Yes\n1.No\nEnter Your Choice: "))
clearConsole()
op = float(input("Old Peak(Range 0.0-5.0):"))
clearConsole()
slp = float(input("Slope:\n0.Nothing\n1.Up Sloping\n2.Flat\n3.Down Sloping\nEnter Your Choice: "))
clearConsole()
nmv = float(input("No. of Major Vessels(Range 0-5): "))
clearConsole()
thmia = float(input("Thalassemia:\n1.Normal\n2.Fixed\n3.Reversable\nEnter Your Choice: "))
clearConsole()


user_data = [age,gender,cpt,rbp,cotl,fbs,recg,mhra,eia,op,slp,nmv,thmia]
result = model.predict(sc.transform([user_data]))

if(result[0]):
    print("Positive, There is a chance you may get Heart Disease.")
else:
    print("Negative, You may not get Heart Disease.")