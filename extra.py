yuan_data = [
    "Eventually", "lost", "remembers", "lovingly", "PERIODT", "YUP",
    "LMFAOOO", "oo", "but", "actually", "its", "not", "3405", "like",
    "that", "at", "all", "watch", "your", "fucking", "mouth"
]

yuan_true = [
    "ENG", "ENG", "ENG", "ENG", "OTH", "ENG",
    "OTH", "FIL", "ENG", "ENG", "ENG", "ENG",
    "OTH", "OTH", "ENG", "ENG", "ENG", "ENG",
    "ENG", "ENG", "ENG"
]

names_data = [
    "Sila", "Sofia", ",", "James", ",", "at", "Amber", "ay", "lumakwatsa", "and", "drank", "tea", "!"
]

names_data = [
    "FIL", "OTH", "OTH", "OTH", "OTH" "FIL", "OTH", "FIL", "ENG", "ENG", "ENG", "ENG", "OTH",
]

models = [BernoulliNB(), ComplementNB(), MultinomialNB(), DecisionTreeClassifier(),  ExtraTreeClassifier()]

for model in models:
    print("Training:", type(model).__name__)
    model.fit(X_train, y_train)

    X_test = get_features(yuan_data)
    preds = model.predict(X_test)

    predictions = []
    for pred in preds:
        if pred in ["ENG-NE", "ENG"]:
            predictions.append("ENG")
        elif pred in ["FIL", "FIL-NE", "FIL-CS"]:
            predictions.append("FIL")
        else:
            predictions.append("OTH")

    for word, label in zip(yuan_data, predictions):
        print(word, "is to", label)  # prints to console

    print("Accuracy:", accuracy_score(yuan_true, predictions))
    print(classification_report(yuan_true, predictions))

    print("\n\n\n")


