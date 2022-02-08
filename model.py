import pandas as pd
import pickle

X = pd.read_csv('CR_Train.csv', encoding="ISO-8859-1")
Y = pd.read_csv('CR_Test.csv', encoding="ISO-8859-1")

YY = Y.values.reshape(119209, )

# Training using Train-Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, YY, test_size=0.2, random_state=20)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg = log_reg.fit(x_test, y_test)
print(log_reg.score(x_test, y_test))

pickle.dump(log_reg, open('log_model.pkl', 'wb'))

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc = svc.fit(x_test, y_test)
print(svc.score(x_test, y_test))

pickle.dump(svc, open('SVM.pkl', 'wb'))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb = nb.fit(x_test, y_test)
print(nb.score(x_test, y_test))

pickle.dump(nb, open('nb.pkl', 'wb'))

