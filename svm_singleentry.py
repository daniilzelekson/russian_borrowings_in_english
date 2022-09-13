from pybor.svm import BagOfSounds
from pybor.evaluate import prf
from svm_experiment import bigrams, trigrams
from my_research_data import wold_english as training
from my_research_data import russian_borrowings as testing


training_bigrams = [[a, bigrams(b), c] for a, b, c in training]
training_trigrams = [[a, trigrams(b), c] for a, b, c in training]

bags_uni = BagOfSounds(training, kernel="linear")
bags_bi = BagOfSounds(training_bigrams, kernel="linear")
bags_tri = BagOfSounds(training_trigrams, kernel="linear")

for entry in testing:
    entry_bigrams = [[a, bigrams(b), c] for a, b, c in [entry, entry]]
    entry_trigrams = [[a, trigrams(b), c] for a, b, c in [entry, entry]]
    print('\n===========================\n[', ''.join(entry[1]), ']')

    print('Bag of sounds SVM unigrams.')
    tests = bags_uni.predict_data([[a, b] for a, b, c in [entry, entry]])
    prec, rec, fs, acc = prf(tests, [entry, entry])
    print("\tAccuracy:  {0:.2f}".format(acc))

    print('Bag of sounds SVM bigrams.')
    tests = bags_bi.predict_data([[a, b] for a, b, c in entry_bigrams])
    prec, rec, fs, acc = prf(tests, entry_bigrams)
    print("\tAccuracy:  {0:.2f}".format(acc))

    print('Bag of sounds SVM trigrams.')
    tests = bags_tri.predict_data([[a, b] for a, b, c in entry_trigrams])
    prec, rec, fs, acc = prf(tests, entry_trigrams)
    print("\tAccuracy:  {0:.2f}".format(acc))


# # # OUTPUT # # #
# ===========================
# [ boʊlʃəvɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ boʊlʃɪ ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ boʊlʃɪvɪst ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ bæbʊʃkə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ bɒlʃəvɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ bɒlʃɪ ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ bɒlʃɪvɪst ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ bəbuːʃkə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ gjulæg ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ gulɑg ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ kɑːməsɑːr ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ kɒmɪsɑː ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ kəlæʃnɪkɑv ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ menʃəvɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ menʃəvɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ mɑːlətɑːf ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ mɒlətɒf ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ noʊmɛŋkləʧʊrə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ perəstrɔɪkə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ perəstrɔɪkə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ poʊɡrəm ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ prɪsɪdiəm ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ prɪsɪdɪəm ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ pɒɡrəm ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ pəgrɑm ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ soʊviət ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ soʊviɛt ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ spʊtnɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ spʌtnɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ səʊviət ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ ædʒɪtprɑːp ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ædʒɪtprɒp ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ æpərætʃɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ɑːpərɑːtʃɪk ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ɛfɛsbi ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ ɛfɛsbiː ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  0.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ɡlæsnɒst ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ɡlæznoʊst ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  1.00
#
# ===========================
# [ ɡuːlæɡ ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ɪntelədʒentsɪə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
#
# ===========================
# [ ɪntelɪdʒentsɪə ]
# Bag of sounds SVM unigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM bigrams.
# 	Accuracy:  1.00
# Bag of sounds SVM trigrams.
# 	Accuracy:  0.00
