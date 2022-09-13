from pybor.markov import DualMarkov, NativeMarkov
import pybor.evaluate as evaluate
from my_research_data import wold_english as training
from my_research_data import target_wordlist as testing


def validate_loan_detection_dual_basis(train_data, test_data, model="kni", smoothing=0.5, order=3):
    dual_model = DualMarkov(train_data, model=model, order=order, smoothing=smoothing)
    print("Dual Markov: Evaluate train dataset.")
    predictions = dual_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    print("Dual Markov: Evaluate test dataset.")
    predictions = dual_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    return dual_model, test_metrics


def validate_loan_detection_native_basis(train_data, test_data, model="kni", smoothing=0.5, order=3, p=0.995):
    native_model = NativeMarkov(train_data, model=model, order=order, smoothing=smoothing, p=p)
    print("Native Markov: Evaluate train dataset.")
    predictions = native_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    print("Native Markov: Evaluate test dataset.")
    predictions = native_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    return native_model, test_metrics


validate_loan_detection_native_basis(training, testing)
validate_loan_detection_dual_basis(training, testing)


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
