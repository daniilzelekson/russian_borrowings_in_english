from pybor.svm import BagOfSounds
from pybor.evaluate import prf, false_positive
from my_research_data import wold_english as training
from my_research_data import target_wordlist as testing


def bigrams(sequence):
    return list(zip(["^"] + sequence[:-1], sequence[1:] + ["$"]))


def trigrams(sequence):
    return list(
        zip(
            ["^", "^"] + sequence[:-1],
            ["^"] + sequence + ["$"],
            sequence[1:] + ["$", "$"],
        )
    )


training_bigrams = [[a, bigrams(b), c] for a, b, c in training]
testing_bigrams = [[a, bigrams(b), c] for a, b, c in testing]

training_trigrams = [[a, trigrams(b), c] for a, b, c in training]
testing_trigrams = [[a, trigrams(b), c] for a, b, c in testing]

print('\nBag of sounds SVM unigrams.')
bags = BagOfSounds(training, kernel="linear")
tests = bags.predict_data([[a, b] for a, b, c in testing])

false_positive(tests, testing, pprint=True)

prec, rec, fs, acc = prf(tests, testing)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))

print('\nBag of sounds SVM bigrams.')
bags = BagOfSounds(training_bigrams, kernel="linear")
tests = bags.predict_data([[a, b] for a, b, c in testing_bigrams])

false_positive(tests, testing_bigrams, pprint=True)

prec, rec, fs, acc = prf(tests, testing_bigrams)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))

print('\nBag of sounds SVM trigrams.')
bags = BagOfSounds(training_trigrams, kernel="linear")
tests = bags.predict_data([[a, b] for a, b, c in testing_trigrams])

false_positive(tests, testing_trigrams, pprint=True)

prec, rec, fs, acc = prf(tests, testing_trigrams)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))


# # # OUTPUT # # #
# Bag of sounds SVM unigrams.
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |  35.00 |    9.00 |      44 |
# | Negatives |  25.00 |    6.00 |      31 |
# | Total     |   0.80 |    0.20 |      75 |
# Precision: 0.80
# Recall:    0.85
# F-Score:   0.82
# Accuracy:  0.80
#
# Bag of sounds SVM bigrams.
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |  19.00 |    8.00 |      27 |
# | Negatives |  26.00 |   22.00 |      48 |
# | Total     |   0.60 |    0.40 |      75 |
# Precision: 0.70
# Recall:    0.46
# F-Score:   0.56
# Accuracy:  0.60
#
# Bag of sounds SVM trigrams.
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |  15.00 |    8.00 |      23 |
# | Negatives |  26.00 |   26.00 |      52 |
# | Total     |   0.55 |    0.45 |      75 |
# Precision: 0.65
# Recall:    0.37
# F-Score:   0.47
# Accuracy:  0.55
