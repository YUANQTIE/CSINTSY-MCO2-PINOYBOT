"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
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
    columns = pandas.read_csv('[CSINTSY] GRP2_MCO2_DataSet.csv')
    word_id = columns['word_id']
    word = columns['word']
    label = columns['label']  
    is_correct = columns['is_correct'] 
    special_tags = columns['special_tags'] 
    corrected_label = columns['corrected_label'] 
    corrected_special_tags = columns['corrected_special_tags'] 
    is_dirty = columns['is_dirty'] 

    english_words = []
    filipino_words = []

    for i in range(0 , len(word)):
        print(word_id[i], " ", word[i], " ", label[i], " ", special_tags[i], " ",  is_correct[i])
        if not is_correct[i]:
            print(corrected_label[i], " ", corrected_special_tags[i])
            if is_dirty[i]:
                print(" IM DIRTY")
        
        if label[i] == "ENG" and is_correct[i]:
            english_words.append(word[i])
        
        if label[i] == "FIL" and is_correct[i]:
            filipino_words.append(word[i])

    print("English Words from data set: ")
        
    for wor in english_words:
        print(wor)
    
    print("Filipino words from data set: ")

    for wor in filipino_words:
        print(wor)

    # word length, letters, substrings as features
    # labels as targets

    #
    #with open('model.pk1', 'rb') as f:
    #    model = pickle.load(f)
    # ^^remove the '#' when model has been trained.

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    return ['FIL' for i in tokens]

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print(tags)