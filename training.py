import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas
import numpy
from features import *

# =================TESTING WORDS==================
random_data = ["@@@@", "GAGO", "KA", "BA", "!!!!!!!!!!", "LOLLLL", "LMAO", "ASF", "WSG", "OTW", "WYA", "SMH", "si", "Nikita", "dragun", "4343255", "dfsfds#44334", "United", "States", "of", "America", "12345", "%%%%%"]

random_label = ["OTH", "FIL", "FIL", "FIL", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "FIL", "OTH", "OTH", "OTH", "OTH", "ENG", "ENG", "ENG", "OTH", "OTH", "OTH"]

random1_data = ["I'm", "so", "tired", "of", "this", "shitty", "weather", "!!!", "Kain", "na", "tayo", "ng", "lunch", "bes", "!", "Hahaha", "LOL", "that", "was", "funny", "as", "hell", ":)"]

random1_label = ["ENG", "ENG", "ENG", "ENG", "ENG", "ENG", "ENG", "OTH", "FIL", "FIL", "FIL", "FIL", "ENG", "OTH", "OTH", "OTH", "OTH", "ENG", "ENG", "ENG", "ENG", "ENG", "OTH"]

yuan_data = [
    "Eventually", "lost", "remembers", "lovingly", "PERIODT", "YUP",
    "LMFAOOO", "oo", "but", "actually", "its", "not", "3405", "like",
    "that", "at", "all", "watch", "your", "fucking", "mouth"
]

yuan_label = [
    "ENG", "ENG", "ENG", "ENG", "OTH", "ENG",
    "OTH", "FIL", "ENG", "ENG", "ENG", "ENG",
    "OTH", "ENG", "ENG", "ENG", "ENG", "ENG",
    "ENG", "ENG", "ENG"
]

names_data = [
    "Sila", "Sofia", ",", "James", ",", "at", "Amber", "ay", "lumakwatsa", "and", "drank", "tea", "!"
]

names_true = [
    "FIL", "OTH", "OTH", "OTH", "OTH", "FIL", "OTH", "FIL", "ENG", "ENG", "ENG", "ENG", "OTH",
]

# =================LOAD DATASET==================
columns = pandas.read_csv('[CSINTSY]GRP2_MCO2_DataSet.csv')
sentence_id = [int(i) for i in columns['sentence_id']]
word_id = [int(i) for i in columns['word_id']]
word = [str(w) for w in columns['word']]
label = [str(w) for w in columns['label']]
special_tags = [str(i) for i in columns['special_tags']]
is_correct = [str(i) for i in columns['is_correct']]
corrected_label = [str(w) for w in columns['corrected_label']]
corrected_special_tags = [str(w) for w in columns['corrected_special_tags']]
is_dirty = [str(w) for w in columns['is_dirty']]

training_data = []
training_targets = []

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

# 70-15-15 train-validation-test
train_data, test_data, train_label, test_label = train_test_split(training_data, training_targets, test_size=0.15, shuffle=False)
train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, test_size=0.15/0.85, shuffle=False)

for i in range(len(test_label)):
    if test_label[i] in ["ENG", "ENG-NE"]:
        test_label[i] = "ENG"
    elif test_label[i] in ["FIL", "FIL-NE", "CS"]:
        test_label[i] = "FIL"
    else:
        test_label[i] = "OTH"

for i in range(len(validation_label)):
    if validation_label[i] in ["ENG", "ENG-NE"]:
        validation_label[i] = "ENG"
    elif validation_label[i] in ["FIL", "FIL-NE", "FIL-CS"]:
        validation_label[i] = "FIL"
    else:
        validation_label[i] = "OTH"

print(len(training_data), len(training_targets))
print(len(train_data), len(test_data), len(validation_data))
print(len(train_label), len(test_label), len(validation_label))

# =================TRAINING AREA==================
features_matrix = []

for i in range(len(train_data)):
    word = train_data[i]
    if i == 0:
        features_matrix.append(get_features(word))
    else:
        features_matrix.append(get_features(word, train_label[i-1]))

X_train = numpy.array(features_matrix)
y_train = numpy.array(train_label)

# =====================TAG CONVERTER==================
for i in range(len(training_targets)):
    if training_targets[i] in ["ENG", "ENG-NE"]:
        training_targets[i] = "ENG"
    elif training_targets[i] in ["FIL", "FIL-NE", "FIL-CS"]:
        training_targets[i] = "FIL"
    else:
        training_targets[i] = "OTH"

# TESTING AREA 
#           ["Yuan Data", yuan_data, yuan_label],
#           ["Names Data", names_data, names_true],
#           ["Test Data", test_data, test_label],
#           ["Validation Data", validation_data, validation_label],
#           ["Random Data", random_data, random_label], 
#           ["Random1 Data", random1_data, random1_label],
#           ["Entire Data", training_data, training_targets]

models = [DecisionTreeClassifier(random_state=2)] #just so fixed yung results

testing = [["Yuan Data", yuan_data, yuan_label],
           ["Names Data", names_data, names_true],
           ["Test Data", test_data, test_label],
           ["Validation Data", validation_data, validation_label],
           ["Random Data", random_data, random_label], 
           ["Random1 Data", random1_data, random1_label],
           ["Entire Data", training_data, training_targets]]

for set_name, data_group, label_group in testing:
    print(len(data_group), len(label_group))
    for model in models:
        test_matrix = []
        model.fit(X_train, y_train)
        with open("trained_model.pk1", "wb") as f:
            pickle.dump(model, f)
        predictions = []
        print("Training on:", set_name, "using", type(model).__name__)
        for i in range(len(data_group)):
            if i == 0:
                prev_tag = None
            else:
                prev_tag = predictions[i-1]

            if is_numeric(data_group[i]) or is_symbolic(data_group[i]) or is_alpha_numeric(data_group[i]):
                predictions.append("OTH")
            elif is_common_english_article(data_group[i]):
                predictions.append("ENG")
            elif is_common_filipino_article(data_group[i]): #checkers that assume the language immediately
                predictions.append("FIL")
            else:

                features = get_features(data_group[i], prev_tag)

                pred = model.predict(numpy.array([features]))[0] #predictor

                if pred in ["ENG", "ENG-ABB"]:
                    pred = "ENG"
                elif pred in ["FIL-ABB", "FIL", "CS"]:
                    pred = "FIL"
                else:
                    pred = "OTH"

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
        ]

        # uncomment this to see actual tags of the word
        #for word, tag, actual in zip(data_group, predictions, label_group): #change params to appropriate data list name and label list name
        #    print(word, "is to", tag, "has to be", actual) 

        if isinstance(model, DecisionTreeClassifier):
            importance_df = pandas.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)
            print(importance_df)

        else:
            log_prob_diff = abs(model.feature_log_prob_[0] - model.feature_log_prob_[1]) #featuer ranker
            importance_df = pandas.DataFrame({
                "feature": feature_names,
                "importance": log_prob_diff
            }).sort_values(by="importance", ascending=False)

        print("Accuracy:", accuracy_score(label_group, predictions))
        print(classification_report(label_group, predictions))

        print("\n\n\n")


