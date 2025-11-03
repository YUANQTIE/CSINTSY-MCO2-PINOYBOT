import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas
import numpy

def has_number(word):
    for char in word:
        if char.isdigit():
            return True
    return False

def has_consecutive_character(word):
    for i in range(1, len(word)):
        if word[i] == word[i-1]:
            return True
    return False

def has_nonfil_character(word):
    word = word.lower()
    for char in word:
        if char == 'c' or char == 'f' or char == 'z' or char == 'j' or char == 'q' or char == 'v' or char == 'x':
            return True
    return False

def get_c_to_v_ratio(word):
    v_count = 1.0
    c_count = 0.0
    word = word.lower()
    for char in word:
        if char >= 'a' and char <= 'z':
            if char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u':
                v_count += 1
            else:
                c_count += 1
    return c_count / v_count

def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

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

print(len(word))

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

print(len(train_data), len(test_data), len(validation_data))
print(len(train_label), len(test_label), len(validation_label))

# feature extraction
# Feature 1: Word Length, longer usually means FIL
# Feature 2: Has soft-sounding affixes, denoting ENG
# Feature 3: Has hard-sounding affixes, denoting FIL
# Feature 4: Has affixes between letters of word, denoting FIL
# Feature 5: Has capitalized starting letter while not being the first word in the sentence, denoting UNK-NE
# Feature 6: Has 3 capitalized letters while being followed by 4 digits, denoting a license plate or UNK-NE
# Feature 7: Consonant to vowel ratio, higher ratio usually means ENG (more consonants than vowels)
# Feature 8: Presence of characters (c, f, z, j, q, v, x), denoting ENG
# Feature 9: Repeated characters in a row, (arrow, mallet), denoting ENG (sometimes its UNK-NE)
# Feature 10: Has numbers but within the word (A4, G6), denoting UNK-NE
# Feature 11: All capitalized characters, usually button mashing (HSAHAHDASHAH or FJEDJOISKLFJDLK) denoting UNK-EXPR
# Feature 12: Higher average syllable length, usually denoting FIL
# feature 13: n-grams

features_matrix = []

for i in range(0, len(train_data)):

    #feature 1
    f1 = len(train_data[i])
    #feature 2
    #feature 3
    #feature 4


    #feature 5

    #checks if first character of word is capitalized, and checks if the current sentence id is between other sentence id's.
    f5 = 0

    if train_data[i][0].isupper() and (sentence_id[i] == sentence_id[i-1] or sentence_id[i] == sentence_id[i+1]):
        f5 = 1

    #feature 6

    f6 = 0

    if i < len(train_data) - 1:
        if len(train_data[i]) == 3 and train_data[i].isupper() and train_data[i+1].isdigit() and 3 <= len(train_data[i+1]) <= 4:
            f6 = 1

    #feature 7

    f7 = get_c_to_v_ratio(train_data[i])

    #feature 8

    f8 = 0
    if has_nonfil_character(train_data[i]):
        f8 = 1

    #feature 9

    f9 = 0
    if has_consecutive_character(train_data[i]):
        f9 = 1

    #feature 10  

    f10 = 0
    if has_number(train_data[i]):
        f10 = 1

    #feature 11  
    
    f11 = 0
    if train_data[i].isupper():
        fil = 1

    #feature 12  

    features_matrix.append([f1, f5, f6, f7, f8, f9, f10, f11])  

x = numpy.array(features_matrix)
y = numpy.array(train_label)

clf = MultinomialNB()
clf.fit(x, y)

def get_features(word):
    f1 = len(word)
    f5 = 1 if word[0].isupper() else 0
    f6 = 0  
    f7 = get_c_to_v_ratio(word)
    f8 = 1 if has_nonfil_character(word) else 0
    f9 = 1 if has_consecutive_character(word) else 0
    f10 = 1 if has_number(word) else 0
    f11 = 1 if word.isupper() else 0
    return [f1, f5, f6, f7, f8, f9, f10, f11]


predictions = []
for word in validation_data:
    if is_numeric(word):
        predictions.append("NUM")
    elif not word.isalnum():
        predictions.append("SYM")
    else:
        word_features = get_features(word)
        word_features = numpy.array(word_features).reshape(1, -1)
        predictions.append(clf.predict(word_features))

for word, label in zip(validation_data, predictions):
    print(word, "is to", label)