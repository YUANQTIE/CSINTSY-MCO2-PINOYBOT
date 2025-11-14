import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas
import numpy
from features import *

# Loads the dataset using pandas csv reader
columns = pandas.read_csv('[CSINTSY]GRP2_MCO2_DataSet.csv')
#sentence_id = [int(i) for i in columns['sentence_id']]
#word_id = [int(i) for i in columns['word_id']]
word = [str(w) for w in columns['word']]
label = [str(w) for w in columns['label']]
special_tags = [str(i) for i in columns['special_tags']]
is_correct = [str(i) for i in columns['is_correct']]
corrected_label = [str(w) for w in columns['corrected_label']]
corrected_special_tags = [str(w) for w in columns['corrected_special_tags']]
is_dirty = [str(w) for w in columns['is_dirty']]

training_data = []
training_targets = []

# For noting the label with the special tag
for isc, w, l, st, cl, cst, id in zip(is_correct, word, label, special_tags, corrected_label, corrected_special_tags, is_dirty):
    if isc:
        if l == "ENG":
            if st == "NE":
                training_targets.append("ENG-NE")
            elif st == "ABB":
                training_targets.append("ENG-ABB")
            elif st == "ABB_NE":
                training_targets.append("ENG-ABB-NE")
            else:
                training_targets.append("ENG")

        elif l == "FIL":
            if st == "NE":
                training_targets.append("FIL-NE")
            elif st == "ABB":
                training_targets.append("FIL-ABB")
            elif st == "ABB_NE":
                training_targets.append("FIL-ABB-NE")
            else:
                training_targets.append("FIL")
        elif l == "UNK":
            if st == "NE":
                training_targets.append("UNK-NE")
            elif st == "EXPR":
                training_targets.append("UNK-EXPR")
            else:
                training_targets.append("UNK")
        elif l == "NUM":
            training_targets.append("NUM")
        else:
            training_targets.append("SYM")
    else:
        if id == "TRUE":
            training_targets.append("UNK")
        else:
            if cl == "ENG":
                if cst == "NE":
                    training_targets.append("ENG-NE")
                elif cst == "ABB":
                    training_targets.append("ENG-ABB")
                elif cst == "ABB_NE":
                    training_targets.append("ENG-ABB-NE")
                else:
                    training_targets.append("ENG")

            elif cl == "FIL":
                if cst == "NE":
                    training_targets.append("FIL-NE")
                elif cst == "ABB":
                    training_targets.append("FIL-ABB")
                elif cst == "ABB_NE":
                    training_targets.append("FIL-ABB-NE")
                else:
                    training_targets.append("FIL")
            elif cl == "UNK":
                if cst == "NE":
                    training_targets.append("UNK-NE")
                elif cst == "EXPR":
                    training_targets.append("UNK-EXPR")
                else:
                    training_targets.append("UNK")
            elif cl == "CS":
                training_targets.append("CS")
            elif cl == "NUM":
                training_targets.append("NUM")
            else:
                training_targets.append("SYM")

    training_data.append(w)

# For noting the expected tag of the training targets
for i in range(len(training_targets)):
    if training_targets[i] in ["ENG", "ENG-NE"]:
        training_targets[i] = "ENG"
    elif training_targets[i] in ["FIL", "FIL-NE", "CS"]:
        training_targets[i] = "FIL"
    else:
        training_targets[i] = "OTH"

# 70-15-15 train-validation-test
train_data, test_data, train_label, test_label = train_test_split(training_data, training_targets, test_size=0.15, shuffle=False)
train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, test_size=0.15/0.85, shuffle=False)

# Preparation for testing by getting the feature matrix of every data
features_matrix = []

for i in range(len(train_data)):
    features_matrix.append(get_features(train_data[i]))


X_train = numpy.array(features_matrix)
y_train = numpy.array(train_label)

# Decision Tree Classifier as the primary classifier to be used
model = DecisionTreeClassifier(max_depth = 30, random_state=2) #specific state to have fixed results

# Feature matrix and classifier combine to make the pipeline
model.fit(X_train, y_train)

# Saving the trained model
with open("trained_model.pk1", "wb") as f:
    pickle.dump(model, f)

# Testing sets used
testing = [["Test Data", test_data, test_label], ["Validation Data", validation_data, validation_label]]

for set_name, data_group, label_group in testing:    
    predictions = []
    for i in range(len(data_group)):

        #Checker for numerical and symbolic words. Automatically assumed to be "OTH"
        if is_numeric(data_group[i]) or is_symbolic(data_group[i]) or is_alpha_numeric(data_group[i]):
            predictions.append("OTH")
        else:
            features = get_features(data_group[i])

            pred = model.predict(numpy.array([features]))[0] #Gets the prediction of the word

            predictions.append(pred)

    feature_names = [ #feature name for importance display
        "Feature 1", 
        "Feature 2", 
        "Feature 3", 
        "Feature 4",
        "Feature 5", 
        "Feature 6", 
        "Feature 7", 
        "Feature 8",
        "Feature 9",
        "Feature 10",
        "Feature 11"
    ]

    # Prints the words that were labeled incorrectly
    for word, tag, actual in zip(data_group, predictions, label_group): 
        if tag != actual:
            print(word, "is to", tag, "has to be", actual) 

    importance_df = pandas.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(by="importance", ascending=False)
    
    # Display of statistics for data analysis
    print(importance_df)
    print(classification_report(label_group, predictions))
    print("\n\n\n")


