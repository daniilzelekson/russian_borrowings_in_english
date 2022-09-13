import pybor.evaluate as evaluate
from pybor.neural import NeuralNative, NeuralDual
from my_research_data import wold_english as training
from my_research_data import target_wordlist as testing


def evaluate_neural_loanword_prediction_train_test(
    language="",
    train=None,
    test=None,
    detect_type="dual",
    model_type="recurrent",
    val_split=None,
    settings=None,
):

    print(f"*** Evaluation of prediction for {language}. ***")
    print(f"Detect type is {detect_type}, neural model type is {model_type}.")
    if detect_type == "native":
        neural = NeuralNative(
            training=train,
            testing=test,
            language=language,
            series="devel",
            model_type=model_type,
            val_split=val_split,
            settings=settings,
        )
    else:
        neural = NeuralDual(
            training=train,
            testing=test,
            language=language,
            series="devel",
            model_type=model_type,
            val_split=val_split,
            settings=settings,
        )

    neural.train()

    print("Evaluate train dataset.")
    predictions = neural.predict_data(train)
    train_metrics = evaluate.evaluate_model(predictions, train)
    evaluate.false_positive(predictions, train)

    if test:
        print("Evaluate test dataset.")
        predictions = neural.predict_data(test)
        test_metrics = evaluate.evaluate_model(predictions, test)
        evaluate.false_positive(predictions, test)


evaluate_neural_loanword_prediction_train_test('English', training, testing, detect_type='dual', model_type='recurrent')


# # # OUTPUT # # #
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
# |       0.786 |    0.805 |     0.795 |      0.773 |
# |           |   True |   False |   Total |
# |:----------|-------:|--------:|--------:|
# | Positives |  33.00 |    9.00 |      42 |
# | Negatives |  25.00 |    8.00 |      33 |
# | Total     |   0.77 |    0.23 |      75 |
