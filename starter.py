import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

'''
Dates: Timestamp of the incident
Category: Category of the incident (target variable)
Descript: Detailed Description of crime incident
DayOfWeek: Day of the week
PdDistrict: Name of Police District
Resolution: How crime was resolved
Address: Approximate street address of crime incident
X - Longitude
Y - Latitude
'''

#Load the data
training = pd.read_csv('data/train.csv', parse_dates = ["Dates"])
testing = pd.read_csv("data/test.csv", parse_dates = ["Dates"])

#Convert Category into numbers
crime_OHE = preprocessing.LabelEncoder()
crime_labels = crime_OHE.fit_transform(training.Category)

#Convert the weekday, district, and hours with OHE
def OHE_crime(df):
	days = pd.get_dummies(df.DayOfWeek)
	district = pd.get_dummies(df.PdDistrict)
	hour = pd.get_dummies(df.Dates.dt.hour)

	new_df = pd.concat([days, hour, district], axis = 1)
	return new_df

#Build new training set
training_OHE = OHE_crime(training)

#Build new testing set
testing_OHE = OHE_crime(testing)

train, train_labels, validation, validation_labels = train_test_split(training_OHE, crime_labels train_size = 0.65)

from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

clf = BernoulliNB()
#clf = LogisticRegression()
#clf= RandomForestClassifier()

clf.fit(train, train_labels)
predicted = np.array(clf.predict_proba(validtion))
log_loss(validation_labels)


#Write results
results_predicted = clf.predict_proba(testing_OHE)
result = pd.DataFrame(results_predicted, columns = le_crime.classes_)
results.to_csv('results/testResults.csv', index = True, index_label = 'Id')




