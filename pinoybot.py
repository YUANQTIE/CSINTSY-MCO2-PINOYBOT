"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import pandas
from features import *


#from sklearn.model_selection import train_test_split
import os
import pickle
import numpy
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
    tags = []
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    with open("trained_model.pk1", "rb") as f:
        model = pickle.load(f)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    for i in range(len(tokens)):
        if i == 0:
            prev_tok = None
        else:
            prev_tok = tokens[i-1]

        if is_numeric(tokens[i]) or is_symbolic(tokens[i]) or is_alpha_numeric(tokens[i]):
            tags.append("OTH")
        elif is_common_english_article(tokens[i]):
            tags.append("ENG")
        elif is_common_filipino_article(tokens[i]): #checkers that assume the language immediately
            tags.append("FIL")
        else:

            features = get_features(tokens[i], prev_tok)

            pred = model.predict(numpy.array([features]))[0] #predictor

            if pred in ["ENG", "ENG-ABB"]:
                pred = "ENG"
            elif pred in ["FIL-ABB", "FIL", "CS"]:
                pred = "FIL"
            else:
                pred = "OTH"

            tags.append(pred)

    

    # 5. Return the list of tags
    #    return tags

    return tags

if __name__ == "__main__":
    # Example usage
    example_tokens = ["I", "didn't", "even", "really", "think", "about", "yung", "repercussions", "nung", "mgapwedeng", "mangyari", "sa", "ginawa", "ni", "Jerome", ".", "mb", ",","akin", "yon", "!"]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)

    print(tags)