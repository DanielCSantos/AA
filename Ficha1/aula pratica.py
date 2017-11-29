# soma da matriz de confusão

#np.sum(GT==results)
#np.sum(GT==3)
#verificar quantos pontos classificados como 1
#np.sum(results(GT==1)==1)


#matriz inversa calculo

#import numpy.linalg as la
#SI1=la.pinv(SI)

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

bc = datasets.load_breast_cancer()

keys = bc.keys()

 x1, x2, t1, t2 = train_teste_split(X, GT, test_size = .5, shuffle = True)

#fit, predict, score----

KNN = KNeighborsClassifier(n_neighbors=5, weights='uniform')

KNN.fit(X1, t1)

estC = KNN.predict(X2)

np.sum(estC! = t2)

KNN.score(X2,t2)

p11 = np.sum(estC[t2 == 0] == 0)

p21 = np.sum(estC[t2 == 1] == 0)
p22 = np.sum(estC[t2 == 1] == 1)