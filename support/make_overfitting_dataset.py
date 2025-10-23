from sklearn.datasets import make_classification
import pandas as pd 
import numpy as np
X, y = make_classification(n_samples=2000, n_features=50, n_informative=5, random_state=523452)
df = pd.DataFrame(data=np.hstack((X, y[:,None])))
df.to_csv("./data/overfitting.csv",index=False)