import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix


def RNN_prediction(classifier, df):
    """Predict text sentiment with a fastai learner.

    Parameters
    ----------
    classifier : [type]
        Text sentiment predictor
    df : pd.DataFrame
        X value is the 'text' column
    """
    # Supress tensorflow warnings FIXME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["predict"] = df["text"].apply(lambda x: classifier.predict(x)[0])
    df["predict"] = df["predict"].astype(int)

    cc_svm = pd.Series(
        confusion_matrix(df["label"], df["predict"]).reshape(1, -1)[0],
        index=["tn", "fp", "fn", "tp"],
    )
    print("\n", cc_svm, "\n")
    print("sensitivity", (cc_svm["tp"] / (cc_svm["tp"] + cc_svm["fn"]).round(3)))
    print("precision", (cc_svm["tp"] / (cc_svm["tp"] + cc_svm["fp"])).round(3))
    print(
        "accuracy",
        (
            (cc_svm["tp"] + cc_svm["tn"])
            / (cc_svm["tp"] + cc_svm["fp"] + cc_svm["fn"] + cc_svm["tn"])
        ).round(3),
    )

    return df
