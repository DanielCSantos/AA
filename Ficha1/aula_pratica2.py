from sklearn.linear_model import LogisticRegression
import numpy as np

LogR = LogisticRegression()
LogR.fit(X1, t1)

LogR.score(X2, t2)

estC = logR.predict(X2)

p12 = np.sum(estC[t2 == 0]==1)
p21 = np.sum(estC[t2 == 1]==0)
