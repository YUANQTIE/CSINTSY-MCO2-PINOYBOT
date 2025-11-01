"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn.model_selection import train_test_split
import os
import pickle
from typing import List

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    columns = pandas.read_csv('[CSINTSY]GRP2_MCO2_DataSet.csv')
    word_id = columns['word_id']
    word = columns['word']
    word = [str(w) for w in word]
    label = columns['label']  
    is_correct = columns['is_correct'] 
    special_tags = columns['special_tags'] 
    corrected_label = columns['corrected_label'] 
    corrected_special_tags = columns['corrected_special_tags'] 
    is_dirty = columns['is_dirty'] 

    english_words = []
    filipino_words = []
    english_NEs = []
    filipino_NEs = []
    english_ABBs = []
    filipino_ABBs = []
    english_ABB_NEs = []
    filipino_ABB_NEs = []
    code_switches = []
    unk_NEs = []
    unk_EXPRs = []

    training_data = []
    training_targets = []

    for i in range(0 , len(word)):

        if is_correct[i]:
            if label[i] == "ENG":
                if special_tags[i] == "NE": 
                    english_NEs.append(word[i])
                    training_targets.append("ENG_NE")
                    training_data.append(word[i])  
                elif special_tags[i] == "ABB_NE":
                    english_ABB_NEs.append(word[i])
                    training_targets.append("ENG_ABB_NE")
                    training_data.append(word[i])  
                elif special_tags[i] == "ABB":
                    english_ABBs.append(word[i])
                    training_targets.append("ENG_ABB")
                    training_data.append(word[i])  
                else:
                    english_words.append(word[i])
                    training_targets.append("ENG")
                    training_data.append(word[i])  
            
            elif label[i] == "FIL":
                if special_tags[i] == "NE": 
                    filipino_NEs.append(word[i])
                    training_targets.append("FIL_NE")
                    training_data.append(word[i])  
                elif special_tags[i] == "ABB_NE":
                    filipino_ABB_NEs.append(word[i])
                    training_targets.append("FIL_ABB_NE")
                    training_data.append(word[i])  
                elif special_tags[i] == "ABB":
                    filipino_ABBs.append(word[i])
                    training_targets.append("FIL_ABB")
                    training_data.append(word[i])  
                else:
                    filipino_words.append(word[i])
                    training_targets.append("FIL")
                    training_data.append(word[i])  
            
            else:
                if special_tags[i] == "EXPR":
                    unk_EXPRs.append(word[i])
                    training_targets.append("UNK_EXPR")
                    training_data.append(word[i])  
                elif special_tags[i] == "NE":
                    unk_NEs.append(word[i])
                    training_targets.append("UNK_NE")
                    training_data.append(word[i])  
                
        
        else:
            if corrected_label[i] == "ENG":
                if corrected_special_tags[i] == "NE": 
                    english_NEs.append(word[i])
                    training_targets.append("ENG_NE")
                    training_data.append(word[i])  
                elif corrected_special_tags[i] == "ABB_NE":
                    english_ABB_NEs.append(word[i])
                    training_targets.append("ENG_ABB_NE")
                    training_data.append(word[i])  
                elif corrected_special_tags[i] == "ABB":
                    english_ABBs.append(word[i])
                    training_targets.append("ENG_ABB")
                    training_data.append(word[i])  
                else:
                    english_words.append(word[i])
                    training_targets.append("ENG")
                    training_data.append(word[i])  
            
            elif corrected_label[i] == "FIL":
                if corrected_special_tags[i] == "NE": 
                    filipino_NEs.append(word[i])
                    training_targets.append("FIL_NE")
                    training_data.append(word[i])  
                elif corrected_special_tags[i] == "ABB_NE":
                    filipino_ABB_NEs.append(word[i])
                    training_targets.append("FIL_ABB_NE")
                    training_data.append(word[i])  
                elif corrected_special_tags[i] == "ABB":
                    filipino_ABBs.append(word[i])
                    training_targets.append("FIL_ABB")
                    training_data.append(word[i])  
                elif corrected_special_tags[i] == "CS":
                    code_switches.append(word[i])
                    training_targets.append("FIL_CS")
                    training_data.append(word[i])  
                else:
                    filipino_words.append(word[i])
                    training_targets.append("FIL")
                    training_data.append(word[i])  
            
            else:
                if corrected_special_tags[i] == "EXPR":
                    unk_EXPRs.append(word[i])
                    training_targets.append("UNK_EXPR")
                    training_data.append(word[i])  
                elif corrected_special_tags[i] == "NE":
                    unk_NEs.append(word[i])
                    training_targets.append("UNK_NE") 
                    training_data.append(word[i])  
    
    print(len(training_data))
    print(len(training_targets))

    
    
    print("English Words: ", english_words)
    #print("\n\nFilipino Words: ",filipino_words)
    print("\n\nEnglish NE Words: ",english_NEs)
    #print("\n\nFilipino Ne Words: ",filipino_NEs)
    #print("\n\nEnglish ABB Words: ",english_ABBs)
    #print("\n\nFilipino ABB Words: ",filipino_ABBs)
    #print("\n\nEnglish ABB NE Words: ",english_ABB_NEs)
    #print("\n\nFilipino ABB NE Words: ",filipino_ABB_NEs)
    #print("\n\nCode Switches: ",code_switches)
    #print("\n\nUNK NE: ",unk_NEs)
    #print("\n\nUNK EXpre: ",unk_EXPRs)

    #
    #with open('model.pk1', 'rb') as f:
    #    model = pickle.load(f)
    # ^^remove the '#' when model has been trained.

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    training_data = [str(w) for w in training_data]

    for d, l in zip(training_data, training_targets):
        print(d, "is to", l)

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(training_data)

    model = MultinomialNB()
    model.fit(matrix, training_targets)

    preds = []

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    for tok in tokens: 
        if not tok.isalpha() or is_numeric(tok): #brute force nalang yung mga symbols and numbers
            preds.append("OTH")
        else:
            X_new = vectorizer.transform([tok])
            prediction = model.predict(X_new)
            print(prediction)


    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    return preds

if __name__ == "__main__":
    # Example usage
    example_tokens = ["flexibility", "see", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)

    print(tags)
