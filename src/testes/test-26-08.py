import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
model = make_pipeline(PolynomialFeatures(2), Ridge())

targets = np.array([[20, 20],
                    [960, 20],
                    [1900, 20],
                    [1900, 540],
                    [960, 540],
                    [20, 540],
                    [20, 1060],
                    [960, 1060],
                    [1900, 1060]])
eyes = np.array([[0.0, 0.0],
                  [0.5, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.5],
                  [0.5, 0.5],
                  [0.0, 0.5],
				  [0.0, 1.0],
                  [0.5, 1.0],
                  [1.0, 1.0]])


print(targets[:, 0])
print(targets[:, 1])
coeffsX = np.linalg.pinv(eyes).dot(targets[:, 0])
coeffsY = np.linalg.pinv(eyes).dot(targets[:, 1])
matrix = np.vstack((coeffsX, coeffsY))

print(matrix.dot(np.array([0.5, 0.5])))


model.fit(eyes, targets[:, 0])
print(model.predict(np.array([[0.5, 0.5]])))