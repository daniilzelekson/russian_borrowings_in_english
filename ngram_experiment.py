from pybor.ngram import NgramModel
from pybor.evaluate import prf, false_positive
from my_research_data import wold_english as training
from my_research_data import target_wordlist as testing


ngram_model = NgramModel(training)
tests = ngram_model.predict_data([[a, b] for a, b, c in testing])

false_positive(tests, testing, pprint=True)

prec, rec, fs, acc = prf(tests, testing)
if all([prec, rec, fs, acc]):
    print("Precision: {0:.2f}".format(prec))
    print("Recall:    {0:.2f}".format(rec))
    print("F-Score:   {0:.2f}".format(fs))
    print("Accuracy:  {0:.2f}".format(acc))


# # # OUTPUT # # #
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |  28.00 |    9.00 |      37 |
# | Negatives |  25.00 |   13.00 |      38 |
# | Total     |   0.71 |    0.29 |      75 |
# Precision: 0.76
# Recall:    0.68
# F-Score:   0.72
# Accuracy:  0.71
