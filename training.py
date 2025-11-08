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
import pandas
import numpy
import math

#English characters
def feature_1(word):
    word = word.lower()
    for char in word:
        if char == 'c' or char == 'f' or char == 'j' or char == 'q' or char == 'v' or char == 'x' or char == 'z':
            return 1
    return 0

#Vowel to word length ratio
def feature_2(word):
    word = word.lower()
    vowel_count = 0
    for char in word:
        if char == "a" or char == "e" or char == "i" or char == "o" or char == "u":
            vowel_count = vowel_count + 1
        
    if math.ceil(vowel_count / len(word)) > 0.35:
        return 1
    else:
        return 0

# 4-letter English n-grams
def feature_3(word):
    word = word.lower()
    if "tion" in word or "ight" in word or "ough" in word or "ment" in word or "less" in word or "sion" in word or "able" in word:
        return 1
    return 0

# 3-letter English n-grams
def feature_4(word):
    word = word.lower()
    if "ity" in word or "ort" in word or "art" in word or "urt" in word:
        return 1
    return 0

# 2-letter English n-grams
def feature_5(word):
    word = word.lower()
    if "ed" in word or "en" in word or "er" in word or "al" in word or "ly" in word or "us" in word or "th" in word or "sh" in word or "ch" in word:
        return 1
    return 0

# 3-letter Filipino n-grams
def feature_6(word):
    word = word.lower()
    if "ang" in word or "nag" in word or "pag" in word or "mag" in word or "nag-" in word or "pag-" in word or "mag-z" in word:
        return 1
    return 0

# 2-letter Filipino n-grams
def feature_7(word):
    word = word.lower()
    if "na" in word or "ma" in word or "pa" in word or "ka" in word or "ng" in word:
        return 1
    return 0

# Consecutive vowels that are the same
def feature_8(word):
    word = word.lower()
    if "aa" in word or "ii" in word or "oo" in word or "uu" in word:
        return 1
    return 0

# Consecutive vowels that are not the same (includes "ee")
def feature_9(word):
    word = word.lower()
    if "ai" in word or "ae" in word or "ao" in word or "au" in word or "ea" in word or "ee" in word or "ei" in word or "eo" in word or "eu" in word or "ia" in word or "ie" in word or "io" in word or "iu" in word or "oa" in word or "oe" in word or "oi" in word or "ou" in word or "ua" in word or "ue" in word or "ui" in word or "uo" in word:
        return 1
    return 0

# Consecutive consonants that are the same
def feature_10(word):
    word = word.lower()
    if "rr" in word or "tt" in word or "pp" in word or "ss" in word or "dd" in word or "rr" in word or "gg" in word or "rr" in word or "ll" in word or "zz" in word or "bb" in word or "mm" in word:
        return 1
    return 0

#Checks if first character is capitalized
def feature_11(word):
    if word and word[0].isupper():
        return 1
    return 0

#Checks if the last two characters of the word are consonants
def feature_12(word):
    vowels = ["a", "e", "i", "o", "u"]
    word = word.lower()
    if len(word) < 2:
        return 0

    last_two = word[len(word)-2:len(word)]
    
    if last_two[0] not in vowels and last_two[1] not in vowels:
        return 1
    return 0

#Checks if all characters are capitalized
def feature_13(word):
    if word.isupper():
        return 1
    return 0

#Checks the word before it
def feature_14(previous_word):
    word_list = ["si", "sa", "ang", "at", "ni"]
    if previous_word in word_list:
        return 1
    return 0

#Checks the word after it
def feature_15(succeeding_word):
    word_list = ["ay", "is", "was", "has"]
    if succeeding_word in word_list:
        return 1
    return 0

def is_numeric(value):
    try:
        float(value)
        return 1
    except ValueError:
        return 0
        
def is_symbolic(word):
    if word.isalpha() or word.isnumeric():
        return 0
    else:
        return 1
    
def get_features(word_list):
    word_features_list = []
    for i, word in enumerate(word_list):
        f1 = feature_1(word)
        f2 = feature_2(word)
        f3 = feature_3(word)
        f4 = feature_4(word)
        f5 = feature_5(word)
        f6 = feature_6(word)
        f7 = feature_7(word)
        f8 = feature_8(word)
        f9 = feature_9(word)
        f10 = feature_10(word)
        f11 = feature_11(word)
        f12 = feature_12(word)
        f13 = feature_13(word)
        if i == 0:
            f14 = 0
        else:
            f15 = feature_14(word)
        if i == len(train_data)-1:
            f15 = 0
        else:
            f15 = feature_15(word)
        f16 = is_numeric(word)
        f17 = is_symbolic(word)
        f18 = len(word)
        
        word_features_list.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18])
    return numpy.array(word_features_list)


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
                elif cst == "CS":
                    training_targets.append("FIL-CS")
                else:
                    training_targets.append("FIL")
            elif cl == "UNK":
                if cst == "NE":
                    training_targets.append("UNK-NE")
                elif cst == "EXPR":
                    training_targets.append("UNK-EXPR")
                else:
                    training_targets.append("UNK")
            elif cl == "NUM":
                training_targets.append("NUM")
            else:
                training_targets.append("SYM")

    training_data.append(w)

# 70-15-15 train-validation-test
train_data, test_data, train_label, test_label = train_test_split(training_data, training_targets, test_size=0.15, random_state=1)
train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label, test_size=0.15/0.85, random_state=1)

for i in range(len(test_label)):
    if test_label[i] in ["ENG", "ENG-NE"]:
        test_label[i] = "ENG"
    elif test_label[i] in ["FIL", "FIL-NE", "FIL-CS"]:
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

print(len(train_data), len(test_data), len(validation_data))
print(len(train_label), len(test_label), len(validation_label))

features_matrix = []
for i, word in enumerate(train_data):
    f1 = feature_1(word)
    f2 = feature_2(word)
    f3 = feature_3(word)
    f4 = feature_4(word)
    f5 = feature_5(word)
    f6 = feature_6(word)
    f7 = feature_7(word)
    f8 = feature_8(word)
    f9 = feature_9(word)
    f10 = feature_10(word)
    f11 = feature_11(word)
    f12 = feature_12(word)
    f13 = feature_13(word)
    if i == 0:
        f14 = 0
    else:
        f15 = feature_14(word)
    if i == len(train_data)-1:
        f15 = 0
    else:
        f15 = feature_15(word)
    f16 = is_numeric(word)
    f17 = is_symbolic(word)
    f18 = len(word)
    
    features_matrix.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18])

X_train = numpy.array(features_matrix)
y_train = numpy.array(train_label)

test_data_true = [
    "OTH", #katotohahannoon
    "OTH", #.
    "FIL", #Layon
    "FIL", #ng
    "OTH", #SMC
    "OTH", #,
    "FIL", #sabi
    "FIL", #ni
    "OTH", #Ang
    "OTH", #,
    "FIL", #na
    "FIL", #linisin
    "FIL", #ang
    "FIL", #mga
    "FIL", #pangunahing
    "FIL", #ilog
    "FIL", #at
    "FIL", #mga
    "FIL", #dinadaluyan
    "FIL", #nito
    "FIL", #na
    "FIL", #barado
    "FIL", #ng
    "FIL", #mga
    "FIL", #basura
    "FIL", #at
    "FIL", #burak
    "FIL", #sa
    "FIL", #nakalipas
    "FIL", #na
    "FIL", #maraming
    "FIL", #dekada
    "FIL", #kung
    "FIL", #kayat
    "FIL", #hindi
    "FIL", #maayos
    "FIL", #ang
    "FIL", #pag-agos
    "FIL", #ng
    "FIL", #tubig
    "FIL", #baha
    "FIL", #patungo
    "FIL", #sa
    "ENG", #Manila
    "ENG", #Bay
    "OTH", #.
    "FIL", #Hindi
    "FIL", #naman
    "FIL", #ito
    "FIL", #winika
    "FIL", #ni
    "OTH", #Hudas
    "FIL", #dahil
    "FIL", #siya
    "FIL", #ay
    "FIL", #nagmamalasakit
    "FIL", #sa
    "FIL", #mga
    "FIL", #dukha
    "OTH", #.
    "FIL", #Ano
    "FIL", #mganda
    "FIL", #gawin
    "FIL", #sa
    "FIL", #mga
    "OTH", #Bbw
    "OTH", #?
    "FIL", #Matatandaang
    "FIL", #nakaraang
    "FIL", #linggo
    "FIL", #ay
    "FIL", #hayagang
    "FIL", #idineklara
    "FIL", #ng
    "FIL", #mga
    "FIL", #kaalyado
    "FIL", #ni
    "OTH", #Cayetano
    "FIL", #ang
    "FIL", #patuloy
    "FIL", #na
    "FIL", #suporta
    "FIL", #sa
    "FIL", #liderato
    "FIL", #nito
    "FIL", #sa
    "FIL", #kabila
    "FIL", #selyadong
    "ENG", #term
    "ENG", #sharing
    "OTH", #.
    "FIL", #Nung
    "OTH", #2019
    "FIL", #nagstart
    "FIL", #kami
    "FIL", #mag-ipon
    "FIL", #ng
    "FIL", #materyales
    "FIL", #para
    "FIL", #sa
    "FIL", #bahay
    "OTH", #,
    "FIL", #nung
    "ENG", #october
    "OTH", #2020
    "FIL", #naman
    "FIL", #nag-umpisa
    "FIL", #na
    "FIL", #sila
    "FIL", #tatay
    "FIL", #sa
    "FIL", #pagpatayo
    "FIL", #nung
    "FIL", #bahay
    "OTH", #,
    "FIL", #dahil
    "FIL", #nga
    "FIL", #di
    "FIL", #naman
    "FIL", #kalakihan
    "FIL", #ang
    "FIL", #kita
    "FIL", #ko
    "FIL", #dito
    "FIL", #sa
    "OTH", #Malaysia
    "FIL", #kaya
    "FIL", #mejo
    "FIL", #mabagal
    "FIL", #ang
    "FIL", #pagpapatayo
    "FIL", #namin
    "FIL", #hanggang
    "FIL", #sa
    "FIL", #pumanaw
    "FIL", #si
    "FIL", #tatay
    "FIL", #nung
    "ENG", #may
    "OTH", #18
    "OTH", #2021
    "OTH", #.
    ]

models = [BernoulliNB(), MultinomialNB(), DecisionTreeClassifier()]

yuan_data = [
    "Eventually", "lost", "remembers", "lovingly", "PERIODT", "YUP",
    "LMFAOOO", "oo", "but", "actually", "its", "not", "3405", "like",
    "that", "at", "all", "watch", "your", "fucking", "mouth"
]

yuan_label = [
    "ENG", "ENG", "ENG", "ENG", "OTH", "ENG",
    "OTH", "FIL", "ENG", "ENG", "ENG", "ENG",
    "OTH", "OTH", "ENG", "ENG", "ENG", "ENG",
    "ENG", "ENG", "ENG"
]

# TESTING AREA

for model in models:
    print("Training:", type(model).__name__)
    model.fit(X_train, y_train)

    X_test = get_features(yuan_data) #input name of data here
    preds = model.predict(X_test)

    predictions = []
    for pred in preds:
        if pred in ["ENG", "ENG-NE"]:
            predictions.append("ENG")
        elif pred in ["FIL", "FIL-NE", "FIL-CS"]:
            predictions.append("FIL")
        else:
            predictions.append("OTH")

    feature_names = [
        "Feature 1", "Feature 2", "Feature 3", "Feature 4",
        "Feature 5", "Feature 6", "Feature 7", "Feature 8",
        "Feature 9", "Feature 10", "Feature 11", "Feature 12",
        "Feature 13", "Feature 14", "Feature 15", "Feature 16",
        "Feature 17", "Feature 18"
    ]

    
    for word, label, actual in zip(yuan_data, predictions, yuan_label): #change params to appropriate data list name and label list name
        print(word, "is to", label, "has to be", actual) 

    if isinstance(model, DecisionTreeClassifier):
        importance_df = pandas.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        print(importance_df)

    else:
        log_prob_diff = abs(model.feature_log_prob_[0] - model.feature_log_prob_[1])
        importance_df = pandas.DataFrame({
            "feature": feature_names,
            "importance": log_prob_diff
        }).sort_values(by="importance", ascending=False)

    print("Accuracy:", accuracy_score(yuan_label, predictions)) #change label name here
    print(classification_report(yuan_label, predictions))

    print("\n\n\n")


