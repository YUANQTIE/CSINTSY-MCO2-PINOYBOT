import pickle
from sklearn.model_selection import train_test_split
import pandas

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

#feature extraction
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
# Feature 11: Al capitalized characters, usually button mashing (HSAHAHDASHAH or FJEDJOISKLFJDLK) denoting UNK-EXPR
# Feature 12: Higher average syllable length, usually denoting FIL

features_matrix = []

for data in train_data:
    #feature 1
    features_matrix.append(len(data))  
    #feature 2
    #feature 3
    #feature 4  
    #feature 5  
    #feature 6  
    #feature 7  
    #feature 8  
    #feature 9  
    #feature 10  
    #feature 11  
    #feature 12  

print(train_data)