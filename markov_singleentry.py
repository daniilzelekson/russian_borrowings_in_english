from markov_experiment import validate_loan_detection_dual_basis, validate_loan_detection_native_basis
from my_research_data import wold_english as training
from my_research_data import russian_borrowings as testing


for entry in testing:
    print('\n===========================\n[', ''.join(entry[1]), ']')
    validate_loan_detection_native_basis(training, [entry, entry])
    validate_loan_detection_dual_basis(training, [entry, entry])


# # # OUTPUT # # #
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.683 |    1.000 |     0.812 |      0.747 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.686 |    0.585 |     0.632 |      0.627 |
#
# ===========================
# [ boʊlʃəvɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ boʊlʃɪ ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ boʊlʃɪvɪst ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ bæbʊʃkə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ bɒlʃəvɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ bɒlʃɪ ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ bɒlʃɪvɪst ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ bəbuːʃkə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ gjulæg ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ gulɑg ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ kɑːməsɑːr ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ kɒmɪsɑː ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ kəlæʃnɪkɑv ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ menʃəvɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ menʃəvɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ mɑːlətɑːf ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ mɒlətɒf ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ noʊmɛŋkləʧʊrə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ perəstrɔɪkə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ perəstrɔɪkə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ poʊɡrəm ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ prɪsɪdiəm ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ prɪsɪdɪəm ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ pɒɡrəm ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ pəgrɑm ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ soʊviət ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ soʊviɛt ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ spʊtnɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ spʌtnɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ səʊviət ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ædʒɪtprɑːp ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ædʒɪtprɒp ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ æpərætʃɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ ɑːpərɑːtʃɪk ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ɛfɛsbi ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ɛfɛsbiː ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ ɡlæsnɒst ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ɡlæznoʊst ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ɡuːlæɡ ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
#
# ===========================
# [ ɪntelədʒentsɪə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
#
# ===========================
# [ ɪntelɪdʒentsɪə ]
# Native Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.996 |    0.811 |     0.894 |      0.908 |
# Native Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# Dual Markov: Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.978 |    1.000 |     0.989 |      0.989 |
# Dual Markov: Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
