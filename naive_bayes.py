#-------------------------------------------------------------------------
# AUTHOR: Jimmy Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

def encode_feature(value, mapping):
    if value not in mapping:
        mapping[value] = len(mapping) - 1
    return mapping[value]

#Reading the training data in a csv file
#--> add your Python code here
X, Y = [], []
outlook_map, temp_map, humid_map, wind_map, play_map = {}, {}, {}, {}, {"Yes": 1, "No": 0}

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        #Transform the original training features to numbers and add them to the 4D array X.
        #For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
        X.append([
            encode_feature(row[1], outlook_map),
            encode_feature(row[2], temp_map),
            encode_feature(row[3], humid_map),
            encode_feature(row[4], wind_map)
        ])
        #Transform the original training classes to numbers and add them to the vector Y.
        #For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        Y.append(play_map[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []
test_instances = []

data_mapping = [outlook_map, temp_map, humid_map, wind_map]

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        test_instances.append(row)
        test_data.append([
            encode_feature(row[1], outlook_map),
            encode_feature(row[2], temp_map),
            encode_feature(row[3], humid_map),
            encode_feature(row[4], wind_map)
        ])

#Printing the header os the solution
#--> add your Python code here
print("Day \tOutlook \tTemperature \tHumidity \tWind \t\tPlayTennis \tConfidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
predictions = clf.predict_proba(test_data)
for i, probs in enumerate(predictions):
    confidence = max(probs)
    if confidence >= 0.75:
        predicted_label = "Yes" if probs[1] > probs[0] else "No"
        print(f"{test_instances[i][0]} \t{test_instances[i][1]} {'\t' if len(test_instances[i][1]) < 8 else ''}\t{test_instances[i][2]} \t\t{test_instances[i][3]} \t\t{test_instances[i][4]} \t\t{predicted_label} \t\t{confidence:.3f}")

