# Import libraries
import pandas as pd
import numpy as np
import datetime

# Today's Date
today = datetime.date.today()

# Read historical data
df = pd.read_csv("Historical_Data.csv")
df['len'] = df['Questions'].str.len()

# Calculate date difference
df['Date'] = pd.to_datetime(df['Date'])
df['date_diff'] = today - df['Date']

# Convert date delta to number of days
df['date_diff'] = (df['date_diff'] / datetime.timedelta(minutes=1))/1440

# Create a list of features
feature_cols = ['Questions','len','date_diff']

# Create dataset for features and target variable
X = df[feature_cols] # Features
y = df.Error # Target variable

# Get one hot encoding of columns B
one_hot = pd.get_dummies(X['Questions'])
# Drop column B as it is now encoded
X = X.drop('Questions',axis = 1)
# Join the encoded df
X = X.join(one_hot)
X

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::,1]

y_pred_proba

# Change output if want to change threshold

THRESHOLD = 0.32
preds = np.where(y_pred_proba > THRESHOLD, 'Yes', 'No')

# Confusion matrix for the output
cnf_matrix = metrics.confusion_matrix(y_test, preds)
cnf_matrix

# Save Model
import pickle

pickle.dump(logreg, open('logreg', 'wb'))