
# Feature 1: Presence of characters that are only found in the English alphabet
def feature_1(word):
    word = word.lower()
    for char in word:
        if char == 'c' or char == 'f' or char == 'j' or char == 'q' or char == 'v' or char == 'x' or char == 'z':
            return 1
    return 0

# Feature 2: Presence of n-grams commonly found in English words
def feature_2(word):
    word = word.lower()
    count = 0
    eng_grams = ["tion", "ight", "ough", "ment", "less", "sion", "able", "ed", "en", "er",
                  "al", "ly", "us", "th", "sh", "ch", "ity", "ort", "art", "urt", "ai", "rr", 
                  "tt", "pp", "ss", "dd", "gg", "ll", "zz", "bb", "mm", "ck", "un"]
    for gram in eng_grams:
        if gram in word:
            count += 1
    return count

# Feature 3: Presence of n-grams commonly found in Filipino words
def feature_3(word):
    word = word.lower()
    count = 0
    eng_grams = ["ang", "ing", "hin", "kan", "tar", "sar", "par", "mag", "pag", "nag", 
                 "na", "ng", "an", "ka", "ma", "ta", "ra", "pa", "sa", "la", "oo", "ii", "aa", "oy", "uy"
                 "ay", "pw", "oy"]
    for gram in eng_grams:
        if gram in word:
            count += 1
    return count

# Feature 4: Checks the 1st letter if capitalized
def feature_4(word):
    return word[0].isupper()

# Feature 5: Checks for presence of vowel digraphs which are common in non-Filipino words
def feature_5(word):
    word = word.lower()
    if "ai" in word or "ae" in word or "ao" in word or "au" in word or "ea" in word or "ee" in word or "ei" in word or "eo" in word or "eu" in word or "ia" in word or "ie" in word or "io" in word or "iu" in word or "oa" in word or "oe" in word or "oi" in word or "ou" in word or "ua" in word or "ue" in word or "ui" in word or "uo" in word:
        return 1
    return 0

# Feature 6: Checks if the last two letters are consonants, which are common in English words
def feature_6(word):
    vowels = ["a", "e", "i", "o", "u"]
    word = word.lower()
    if len(word) < 2:
        return 0

    last_two = word[len(word)-2:len(word)]
    
    if last_two[0] not in vowels and last_two[1] not in vowels:
        return 1
    return 0

# Feature 7: Counts the number of uppercase letters in the word for anticipating EXPRs or ABBs
def feature_7(word):
    count = 0
    for char in word:
        if char.isupper():
            count += 1
    return count

# Feature 8: Ratio of vowels to word length. Higher ratio would indicate Filipino word
def feature_8(word):
    count = 0
    word = word.lower()
    vowels = ["a", "e", "i", "o", "u"]
    for char in word:
        if char in vowels:
            count += 1
    
    return count / len(word)
    
# Feature 9: Ratio of consonants to word length. Higher ratio would indicate English word
def feature_9(word):
    count = 0
    word = word.lower()
    vowels = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"]
    for char in word:
        if char in vowels:
            count += 1
    
    return count / len(word)

# Feature 10: Maximum number of occurrences of a single alphabet character in the word used to anticipate repeated characters in EXPRs or ABBs
def feature_10(word):
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    encountered = [0] * 26
    
    for char in word:
        if char in alphabet:
            index = alphabet.index(char)
            encountered[index] += 1

    return max(encountered)

# Checker for numeric tokens
def is_numeric(value):
    try:
        float(value)
        return 1
    except ValueError:
        return 0

# Checker for symbolic tokens 

def is_symbolic(word):
    for char in word:
        if char.isalpha():
            return 0
        
    if word.isalpha() or word.isnumeric():
        return 0
    else:
        return 1

# Checker for tokens that have both alphabetic and numeric characters (assumed to always be NE)
def is_alpha_numeric(word):
    char_present, digit_present = False, False
    for char in word:
        if char.isalpha():
            char_present = True
        elif char.isdigit():
            digit_present = True
    
    return char_present and digit_present

# Checker for common English articles and pronouns
def is_common_english_article(word):
    word = word.lower()
    common_articles = ["i", "i'm", "she", "he", "not" "like", "you", "him", "here", "here", "they", "he", "the", "a", "an", "and", "but", "or", "for", "nor", "so", "yet", "by", "in", "of", "on", "to", "up", "as", "is", "it", "was", "are", "that", "there", "this"]
    
    if word in common_articles:
        return True
    else:
        return False
    
# Checker for common English articles and pronouns
def is_common_filipino_article(word):
    word = word.lower()
    common_articles = ["tsaka", "sa", "at", "mga", "siya", "ikaw", "ni", "kay", "kaysa", "dahil", "upang", "para", "o", "pero", "subalit", "kung", "kapag", "habang", "dapat", "ang"]
    
    if word in common_articles:
        return True
    else:
        return False
    
def get_features(word, previous_word=None):
    f1 = feature_1(word)
    f2 = feature_2(word)
    f3 = feature_3(word)
    f4 = feature_4(word)
    f5 = feature_5(word)
    f6 = feature_6(word)
    f7 = feature_7(word)
    f8 = feature_8(word)
    f9 = feature_9(word)
    #f10 = feature_10(word)
    #f11 = feature_11(word)
    #f12 = feature_12(word)
    #f13 = feature_13(word)
    #f14 = feature_14(word)
    #f15 = feature_15(word)
    
    features = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    return features
