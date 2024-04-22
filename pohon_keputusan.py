from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# # Mount google drive
# drive.mount('/content/drive', force_remount=True)
# from google.colab import drive

# # Import libraries
# from os import chdir
# chdir('/content/drive/My Drive/_OPRIBADI/BUKU_MLSPK/')

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load data
data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

# Preprocess data
X_train_awal = data_train.drop('hasil', axis=1)
X_train = np.array(X_train_awal.values)
y_train_awal = data_train['hasil']
y_train = np.array(y_train_awal.values)

X_test_awal = data_test.drop('hasil', axis=1)
X_test = pd.DataFrame(X_test_awal.values)
y_test_awal = data_test['hasil']
y_test = pd.DataFrame(y_test_awal.values)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
X_test_prediction = clf.predict(X_test)
X_test_predictionl = clf.predict(X_train)

# Evaluate model
X_test_awal['label_asli'] = y_test_awal
X_test_awal['label_pred'] = X_test_prediction
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print(test_data_accuracy)