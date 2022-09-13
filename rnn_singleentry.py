from rnn_experiment import evaluate_neural_loanword_prediction_train_test
from my_research_data import wold_english as training
from my_research_data import russian_borrowings as testing

for entry in testing:
    print('\n===========================\n[', ''.join(entry[1]), ']')
    evaluate_neural_loanword_prediction_train_test('English', train=training, test=[entry, entry])


# # # OUTPUT # # #
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.954 |    0.924 |     0.939 |      0.943 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   14.00 |     307 |
# | Negatives | 334.00 |   24.00 |     358 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.791 |    0.829 |     0.810 |      0.787 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |  34.00 |    9.00 |      43 |
# | Negatives |  25.00 |    7.00 |      32 |
# | Total     |   0.79 |    0.21 |      75 |
#
# ===========================
# [ boʊlʃəvɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.955 |    0.931 |     0.942 |      0.946 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 295.00 |   14.00 |     309 |
# | Negatives | 334.00 |   22.00 |     356 |
# | Total     |   0.95 |    0.05 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ boʊlʃɪ ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.949 |    0.937 |     0.943 |      0.946 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 297.00 |   16.00 |     313 |
# | Negatives | 332.00 |   20.00 |     352 |
# | Total     |   0.95 |    0.05 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   0.00 |    0.00 |       0 |
# | Negatives |   0.00 |    2.00 |       2 |
# | Total     |   0.00 |    1.00 |       2 |
#
# ===========================
# [ boʊlʃɪvɪst ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.934 |    0.934 |     0.934 |      0.937 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   21.00 |     317 |
# | Negatives | 327.00 |   21.00 |     348 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ bæbʊʃkə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.942 |    0.924 |     0.933 |      0.937 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   18.00 |     311 |
# | Negatives | 330.00 |   24.00 |     354 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ bɒlʃəvɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.948 |    0.924 |     0.936 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   16.00 |     309 |
# | Negatives | 332.00 |   24.00 |     356 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ bɒlʃɪ ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.939 |    0.924 |     0.932 |      0.935 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   19.00 |     312 |
# | Negatives | 329.00 |   24.00 |     353 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   0.00 |    0.00 |       0 |
# | Negatives |   0.00 |    2.00 |       2 |
# | Total     |   0.00 |    1.00 |       2 |
#
# ===========================
# [ bɒlʃɪvɪst ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.954 |    0.918 |     0.936 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 291.00 |   14.00 |     305 |
# | Negatives | 334.00 |   26.00 |     360 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ bəbuːʃkə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.935 |    0.915 |     0.925 |      0.929 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 290.00 |   20.00 |     310 |
# | Negatives | 328.00 |   27.00 |     355 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ gjulæg ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.945 |    0.915 |     0.929 |      0.934 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 290.00 |   17.00 |     307 |
# | Negatives | 331.00 |   27.00 |     358 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ gulɑg ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.963 |    0.912 |     0.937 |      0.941 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 289.00 |   11.00 |     300 |
# | Negatives | 337.00 |   28.00 |     365 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   0.00 |    0.00 |       0 |
# | Negatives |   0.00 |    2.00 |       2 |
# | Total     |   0.00 |    1.00 |       2 |
#
# ===========================
# [ kɑːməsɑːr ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.958 |    0.934 |     0.946 |      0.949 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   13.00 |     309 |
# | Negatives | 335.00 |   21.00 |     356 |
# | Total     |   0.95 |    0.05 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ kɒmɪsɑː ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.942 |    0.931 |     0.937 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 295.00 |   18.00 |     313 |
# | Negatives | 330.00 |   22.00 |     352 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ kəlæʃnɪkɑv ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.942 |    0.915 |     0.928 |      0.932 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 290.00 |   18.00 |     308 |
# | Negatives | 330.00 |   27.00 |     357 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ menʃəvɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.951 |    0.915 |     0.932 |      0.937 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 290.00 |   15.00 |     305 |
# | Negatives | 333.00 |   27.00 |     360 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ menʃəvɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.952 |    0.934 |     0.943 |      0.946 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   15.00 |     311 |
# | Negatives | 333.00 |   21.00 |     354 |
# | Total     |   0.95 |    0.05 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ mɑːlətɑːf ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.949 |    0.934 |     0.941 |      0.944 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   16.00 |     312 |
# | Negatives | 332.00 |   21.00 |     353 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   0.00 |    0.00 |       0 |
# | Negatives |   0.00 |    2.00 |       2 |
# | Total     |   0.00 |    1.00 |       2 |
#
# ===========================
# [ mɒlətɒf ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.940 |    0.934 |     0.937 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   19.00 |     315 |
# | Negatives | 329.00 |   21.00 |     350 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ noʊmɛŋkləʧʊrə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.961 |    0.921 |     0.940 |      0.944 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 292.00 |   12.00 |     304 |
# | Negatives | 336.00 |   25.00 |     361 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ perəstrɔɪkə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.940 |    0.934 |     0.937 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   19.00 |     315 |
# | Negatives | 329.00 |   21.00 |     350 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ perəstrɔɪkə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.942 |    0.931 |     0.937 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 295.00 |   18.00 |     313 |
# | Negatives | 330.00 |   22.00 |     352 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ poʊɡrəm ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.945 |    0.921 |     0.933 |      0.937 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 292.00 |   17.00 |     309 |
# | Negatives | 331.00 |   25.00 |     356 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ prɪsɪdiəm ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.942 |    0.931 |     0.937 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 295.00 |   18.00 |     313 |
# | Negatives | 330.00 |   22.00 |     352 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ prɪsɪdɪəm ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.938 |    0.912 |     0.925 |      0.929 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 289.00 |   19.00 |     308 |
# | Negatives | 329.00 |   28.00 |     357 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ pɒɡrəm ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.949 |    0.937 |     0.943 |      0.946 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 297.00 |   16.00 |     313 |
# | Negatives | 332.00 |   20.00 |     352 |
# | Total     |   0.95 |    0.05 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ pəgrɑm ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.943 |    0.937 |     0.940 |      0.943 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 297.00 |   18.00 |     315 |
# | Negatives | 330.00 |   20.00 |     350 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ soʊviət ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.952 |    0.931 |     0.941 |      0.944 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 295.00 |   15.00 |     310 |
# | Negatives | 333.00 |   22.00 |     355 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ soʊviɛt ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.948 |    0.927 |     0.938 |      0.941 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 294.00 |   16.00 |     310 |
# | Negatives | 332.00 |   23.00 |     355 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ spʊtnɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.934 |    0.940 |     0.937 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 298.00 |   21.00 |     319 |
# | Negatives | 327.00 |   19.00 |     346 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   0.00 |    0.00 |       0 |
# | Negatives |   0.00 |    2.00 |       2 |
# | Total     |   0.00 |    1.00 |       2 |
#
# ===========================
# [ spʌtnɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.948 |    0.912 |     0.929 |      0.934 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 289.00 |   16.00 |     305 |
# | Negatives | 332.00 |   28.00 |     360 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ səʊviət ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.936 |    0.921 |     0.928 |      0.932 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 292.00 |   20.00 |     312 |
# | Negatives | 328.00 |   25.00 |     353 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ædʒɪtprɑːp ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.949 |    0.934 |     0.941 |      0.944 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   16.00 |     312 |
# | Negatives | 332.00 |   21.00 |     353 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ædʒɪtprɒp ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.960 |    0.912 |     0.935 |      0.940 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 289.00 |   12.00 |     301 |
# | Negatives | 336.00 |   28.00 |     364 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ æpərætʃɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.939 |    0.927 |     0.933 |      0.937 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 294.00 |   19.00 |     313 |
# | Negatives | 329.00 |   23.00 |     352 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɑːpərɑːtʃɪk ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.957 |    0.912 |     0.934 |      0.938 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 289.00 |   13.00 |     302 |
# | Negatives | 335.00 |   28.00 |     363 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɛfɛsbi ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.939 |    0.931 |     0.935 |      0.938 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 295.00 |   19.00 |     314 |
# | Negatives | 329.00 |   22.00 |     351 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɛfɛsbiː ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.958 |    0.924 |     0.941 |      0.944 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   13.00 |     306 |
# | Negatives | 335.00 |   24.00 |     359 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |           0 |    0.000 |         0 |      0.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   0.00 |    0.00 |       0 |
# | Negatives |   0.00 |    2.00 |       2 |
# | Total     |   0.00 |    1.00 |       2 |
#
# ===========================
# [ ɡlæsnɒst ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.933 |    0.927 |     0.930 |      0.934 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 294.00 |   21.00 |     315 |
# | Negatives | 327.00 |   23.00 |     350 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɡlæznoʊst ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.951 |    0.924 |     0.938 |      0.941 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   15.00 |     308 |
# | Negatives | 333.00 |   24.00 |     357 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɡuːlæɡ ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.952 |    0.934 |     0.943 |      0.946 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 296.00 |   15.00 |     311 |
# | Negatives | 333.00 |   21.00 |     354 |
# | Total     |   0.95 |    0.05 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɪntelədʒentsɪə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.954 |    0.924 |     0.939 |      0.943 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 293.00 |   14.00 |     307 |
# | Negatives | 334.00 |   24.00 |     358 |
# | Total     |   0.94 |    0.06 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
#
# ===========================
# [ ɪntelɪdʒentsɪə ]
# *** Evaluation of prediction for English. ***
# Detect type is dual, neural model type is recurrent.
# Evaluate train dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       0.936 |    0.921 |     0.928 |      0.932 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives | 292.00 |   20.00 |     312 |
# | Negatives | 328.00 |   25.00 |     353 |
# | Total     |   0.93 |    0.07 |     665 |
# Evaluate test dataset.
# |   Precision |   Recall |   F-score |   Accuracy |
# |------------:|---------:|----------:|-----------:|
# |       1.000 |    1.000 |     1.000 |      1.000 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |   2.00 |    0.00 |       2 |
# | Negatives |   0.00 |    0.00 |       0 |
# | Total     |   1.00 |    0.00 |       2 |
