
import numpy as np
from sklearn.linear_model import LinearRegression
X= np.array([[1],[2],[3],[4]])
y=[3,5,7,9]
reg=LinearRegression().fit(X,y)
reg.score(X,y)

print(reg.coef_)

print(reg.intercept_)

reg.predict(np.array([[5]]))

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X= np.array([[1,1],[1,2],[2,2],[2,3]])
y=[6,8,9,11]
reg=LinearRegression().fit(X,y)
print(reg.score(X,y))
print(reg.coef_)
print(reg.intercept_)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y= datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis,2]
diabetes_X_train=diabetes_X[:-20]
diabetes_X_test=diabetes_X[-20:]
diabetes_y_train=diabetes_y[:-20]
diabetes_y_test=diabetes_y[-20:]
reg= linear_model.LinearRegression()
reg.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_pred= reg.predict(diabetes_X_test)
print('coefficient: \n', reg.coef_)
print('Mean Squared error: %.2f'  % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Coefficient of determination: %.2f'   % r2_score(diabetes_y_test,diabetes_y_pred))

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.show()