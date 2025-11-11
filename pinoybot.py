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
    with open("trained_model.pk1", "rb") as f:
        model = pickle.load(f)

    tags = []

    for i in range(len(tokens)):
        if is_numeric(tokens[i]) or is_symbolic(tokens[i]) or is_alpha_numeric(tokens[i]):
            tags.append("OTH")
        else:
            if i == 0:
                prev_word = None
            else:
                prev_word = tags[i-1]
            features = get_features(tokens[i], prev_word)
            pred = model.predict(numpy.array([features]))[0] 
            tags.append(pred)

    tags = [str(tag) for tag in tags]
    return tags

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)

    print(tags)