from pybor.ngram import NgramModel
from pybor.evaluate import prf
from my_research_data import wold_english as training
from my_research_data import russian_borrowings as testing

for entry in testing:
    print('\n[', ''.join(entry[1]), ']')
    ngram_model = NgramModel(training)
    tests = ngram_model.predict_data([[a, b] for a, b, c in [entry, entry]])
    prec, rec, fs, acc = prf(tests, [entry, entry])
    print("Accuracy:  {0:.2f}".format(acc))


# # # OUTPUT # # #
# [ boʊlʃəvɪk ]
# Accuracy:  1.00
#
# [ boʊlʃɪ ]
# Accuracy:  0.00
#
# [ boʊlʃɪvɪst ]
# Accuracy:  0.00
#
# [ bæbʊʃkə ]
# Accuracy:  0.00
#
# [ bɒlʃəvɪk ]
# Accuracy:  1.00
#
# [ bɒlʃɪ ]
# Accuracy:  0.00
#
# [ bɒlʃɪvɪst ]
# Accuracy:  0.00
#
# [ bəbuːʃkə ]
# Accuracy:  1.00
#
# [ gjulæg ]
# Accuracy:  0.00
#
# [ gulɑg ]
# Accuracy:  0.00
#
# [ kɑːməsɑːr ]
# Accuracy:  1.00
#
# [ kɒmɪsɑː ]
# Accuracy:  1.00
#
# [ kəlæʃnɪkɑv ]
# Accuracy:  1.00
#
# [ menʃəvɪk ]
# Accuracy:  1.00
#
# [ menʃəvɪk ]
# Accuracy:  1.00
#
# [ mɑːlətɑːf ]
# Accuracy:  1.00
#
# [ mɒlətɒf ]
# Accuracy:  1.00
#
# [ noʊmɛŋkləʧʊrə ]
# Accuracy:  0.00
#
# [ perəstrɔɪkə ]
# Accuracy:  1.00
#
# [ perəstrɔɪkə ]
# Accuracy:  1.00
#
# [ poʊɡrəm ]
# Accuracy:  1.00
#
# [ prɪsɪdiəm ]
# Accuracy:  1.00
#
# [ prɪsɪdɪəm ]
# Accuracy:  1.00
#
# [ pɒɡrəm ]
# Accuracy:  1.00
#
# [ pəgrɑm ]
# Accuracy:  1.00
#
# [ soʊviət ]
# Accuracy:  1.00
#
# [ soʊviɛt ]
# Accuracy:  1.00
#
# [ spʊtnɪk ]
# Accuracy:  0.00
#
# [ spʌtnɪk ]
# Accuracy:  0.00
#
# [ səʊviət ]
# Accuracy:  1.00
#
# [ ædʒɪtprɑːp ]
# Accuracy:  1.00
#
# [ ædʒɪtprɒp ]
# Accuracy:  1.00
#
# [ æpərætʃɪk ]
# Accuracy:  1.00
#
# [ ɑːpərɑːtʃɪk ]
# Accuracy:  1.00
#
# [ ɛfɛsbi ]
# Accuracy:  0.00
#
# [ ɛfɛsbiː ]
# Accuracy:  0.00
#
# [ ɡlæsnɒst ]
# Accuracy:  1.00
#
# [ ɡlæznoʊst ]
# Accuracy:  1.00
#
# [ ɡuːlæɡ ]
# Accuracy:  0.00
#
# [ ɪntelədʒentsɪə ]
# Accuracy:  1.00
#
# [ ɪntelɪdʒentsɪə ]
# Accuracy:  1.00
