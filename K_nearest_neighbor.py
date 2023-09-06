from sklearn.preprocessing import StandardScaler
import pandas as pd
from package_KNN import Knn
from sklearn.metrics import accuracy_score
import numpy as np
df = pd.read_csv('gene_expression.csv')
print(len(df))
output = df['Cancer Present'][:2400]
input_data = df.drop('Cancer Present', axis=1)[:2400]
#test_data
test_data=df.drop('Cancer Present', axis=1)[2400:]
y_true=df['Cancer Present'][2400:]

test_knn = Knn(7, output, input_data,test_data)

y_pred = test_knn.predict()
print(y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Accuracy:", accuracy)

