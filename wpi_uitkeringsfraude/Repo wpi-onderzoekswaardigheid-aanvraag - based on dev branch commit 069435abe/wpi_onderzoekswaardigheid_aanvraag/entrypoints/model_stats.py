"""Script to generate model statistics such as baseline and lift curve."""
from __future__ import annotations

import argparse
import logging
import os
from importlib.resources import open_binary
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_errors  # noqa: F401
from azureml.core import Run
from fraude_preventie.evaluation.sklearn_predictor_type import SklearnPredictor
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from wpi_onderzoekswaardigheid_aanvraag.model.build_model import (
    filter_application_handling,
    split_data_train_test,
)

logger = logging.getLogger("wpi_onderzoekswaardigheid_aanvraag.model_stats")
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

output = """
# General

The model is a **{model_class}** model.

The **training flags** were: {model_flags:s}.

The **training steps** were: {training_steps}.

The feature selection method was: {feature_selection_method}

{cut_fimp_statement}

The list of features is attached at the bottom of this document.

The final parameter settings were:

{params}


# Metrics

## Training data

| Metric | Value |
|--------|-------|
| Nr of applications | {tr_n_applications:d} |
| Nr of HH onderzoeken | {tr_n_hh_onderzoek:d} |
| Nr of HH screenings | {tr_n_hh_screening:d} |
| Nr of IC screenings | {tr_n_ic_screening:d} |
| Hit rate without model | {tr_frac_pos:.1%} |
| Fraction sent to HH by model (threshold=0.5) | {tr_support:.1%} |
| Hit rate with model (threshold=0.5) | {tr_precision:.1%} |
| Lift (threshold=0.5) | {tr_lift:.1%} |
| AUC | {tr_auc:.2f} |

Notes:
- The hit rate indicates how many of the applications sent to HH were actually onderzoekswaardig.
- A lift of e.g. 1.3 means that the hit rate of the model is 30% higher than when using no model.


## Test data

| Metric | Value|
|--------|------|
| Nr of applications | {te_n_applications:d} |
| Nr of HH onderzoeken | {te_n_hh_onderzoek:d} |
| Nr of HH screenings | {te_n_hh_screening:d} |
| Nr of IC screenings | {te_n_ic_screening:d} |
| Hit rate without model | {te_frac_pos:.1%} |
| Fraction sent to HH by model (threshold=0.5) | {te_support:.1%} |
| Hit rate with model (threshold=0.5) | {te_precision:.1%} |
| Lift (threshold=0.5) | {te_lift:.1%} |
| AUC | {te_auc:.2f} |

Notes:
- The hit rate indicates how many of the applications sent to HH were actually onderzoekswaardig.
- A lift of e.g. 1.3 means that the hit rate of the model is 30% higher than when using no model.

### Confusion matrix
{te_confusion_matrix}


# Feature importance

In the trained model, the selected features contribute to each prediction as shown in
the following chart:

![model feature importance](./feature_importances.png)


# Lift curve

The lift curve shows the increase in precision relative to the baseline number of positives in the dataset when all
 applications above a certain score threshold are sent to HH. As a secondary metric the plot shows the fraction of
 data that is above the given threshold.

## Training data

![lift chart train](./lift_curve_train.png)

## Test data

![lift chart test](./lift_curve_test.png)


# ROC curve

The ROC curve shows the tradeoff between the True Positive Rate (TPR) and False Positive Rate (FPR) at different
 score thresholds, where:

- TPR = recall = sensitivity = TP / P = TP / (TP + FN)
- FPR = 1 - specificity = FP / N = FP / (TN + FP)

## Training data

![roc plot train](./roc_curve_train.png)

## Test data

![roc plot test](./roc_curve_test.png)


# Features

The model has been trained on the following features:
{mapped_feature_names}

Before data prep the feature set was:
{feature_names}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate model statistics.",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Location of the trained model. Default: Use model packaged with the wpi_onderzoekswaardigheid_aanvraag package.",
    )
    parser.add_argument(
        "--data",
        help="File containing pickled output of master pipeline.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default="data/stats",
        help="Output directory where statistics and plots will be stored.",
    )

    args = parser.parse_args()
    model_file = args.model
    data_file = args.data
    output_dir = Path(args.output_dir)
    create_output_dir(output_dir)

    model_dict = load_model(model_file)
    data = load_data(data_file)
    data = filter_application_handling(data, model_dict["model"].handling_types)

    X_train, X_test, y_train, y_test = split_data_train_test(
        data, shuffle=True, label="onderzoekswaardig", random_state=42
    )

    X_test["onderzoekswaardig"] = y_test
    X_train["onderzoekswaardig"] = y_train

    generate_statistics_markdown(X_train, X_test, model_dict, output_dir)


def generate_statistics_markdown(train_data, test_data, model_dict):
    model = model_dict["model"]

    run = Run.get_context()
    mounted_output_dir = run.output_datasets["statistics"]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)

    # Test data
    applications = test_data
    te_data_stats = analyze_data(applications)
    te_data_stats = {f"te_{k}": v for k, v in te_data_stats.items()}
    predictions = score(applications, model)
    baseline = te_data_stats["te_frac_pos"]
    te_performance = calculate_model_performance(
        applications["onderzoekswaardig"],
        predictions["prediction"],
        predictions["score_onderzoekswaardig"],
    )
    run = Run.get_context()
    for name, performance in te_performance.items():
        run.log(name, performance)
    te_performance = {f"te_{k}": v for k, v in te_performance.items()}
    lc = calculate_lift_curve(
        applications["onderzoekswaardig"],
        predictions["score_onderzoekswaardig"],
        baseline,
        score_levels=np.arange(0, 1.0001, 0.05),
    )
    plot_lift_curve(lc)
    plt.savefig(f"{mounted_output_dir}/lift_curve_test.png", dpi=80)
    plot_roc_curve(
        applications["onderzoekswaardig"], predictions["score_onderzoekswaardig"]
    )
    plt.savefig(f"{mounted_output_dir}/roc_curve_test.png", dpi=80)

    # Train data
    applications = train_data
    tr_data_stats = analyze_data(applications)
    tr_data_stats = {f"tr_{k}": v for k, v in tr_data_stats.items()}
    predictions = score(applications, model)
    baseline = tr_data_stats["tr_frac_pos"]
    tr_performance = calculate_model_performance(
        applications["onderzoekswaardig"],
        predictions["prediction"],
        predictions["score_onderzoekswaardig"],
    )
    tr_performance = {f"tr_{k}": v for k, v in tr_performance.items()}
    lc = calculate_lift_curve(
        applications["onderzoekswaardig"],
        predictions["score_onderzoekswaardig"],
        baseline,
        score_levels=np.arange(0, 1.0001, 0.05),
    )
    plot_lift_curve(lc)
    plt.savefig(f"{mounted_output_dir}/lift_curve_train.png", dpi=80)
    plot_roc_curve(
        applications["onderzoekswaardig"], predictions["score_onderzoekswaardig"]
    )
    plt.savefig(f"{mounted_output_dir}/roc_curve_train.png", dpi=80)

    # General
    plot_model_feature_importance(model_dict, n_feats=20)
    plt.savefig(f"{mounted_output_dir}/feature_importances.png", dpi=80)

    model_class = model.named_steps["clf"].best_estimator_.__class__.__name__
    feature_names = model.raw_feature_names
    mapped_feature_names = model.mapped_feature_names
    training_steps = [s for s in model.named_steps]
    flags = model_dict["flags"]
    params_df = pd.DataFrame.from_dict(
        model.named_steps["clf"].best_params_,
        orient="index",
        columns=["Value"],
    )
    params_df = (
        params_df.reset_index()
        .rename(columns={"index": "Parameter"})
        .to_markdown(index=False)
    )

    if model_dict["feature_selection_method"] == "cut_fimp":
        try:
            cut_fimp_statement = _generate_cut_fimp_statement(
                model_dict["cum_fimp_threshold"],
                len(feature_names),
                model_dict["n_original_features"],
            )
        except KeyError:
            cut_fimp_statement = _generate_cut_fimp_statement(
                model_dict["cum_fimp_threshold"], len(feature_names)
            )
    elif model_dict["feature_selection_method"] == "forward_feature_selection":
        cut_fimp_statement = f"Forward feature selection was used. {len(feature_names)} features were selected."
    else:
        cut_fimp_statement = ""

    text = output.format(
        **tr_data_stats,
        **te_data_stats,
        **tr_performance,
        **te_performance,
        model_class=model_class,
        model_flags=str(flags),
        training_steps=training_steps,
        feature_names=list_to_markdown(feature_names),
        mapped_feature_names=list_to_markdown(mapped_feature_names),
        params=params_df,
        cut_fimp_statement=cut_fimp_statement,
        feature_selection_method=model_dict["feature_selection_method"],
    )

    with open(f"{mounted_output_dir}/statistics.md", "w") as f:
        f.write(text)
    logger.info(text)
    logger.info(
        f"The statistics have been written to {mounted_output_dir}/info/model_stats/statistics.md"
    )

    return text


def list_to_markdown(items):
    """Convert to markdown list"""
    return "\n".join([f"- {n}" for n in items])


def plot_lift_curve(lc):
    fig, ax = plt.subplots(1, 1)
    ax.plot(lc.confidence, lc.lift, color="blue", marker="o")
    ax.grid()
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Lift", color="blue")
    ax.axhline(1, color="black", linestyle="--", label="baseline")
    ax2 = ax.twinx()
    ax2.plot(lc.confidence, lc.pct_data, color="orange", marker="o")
    ax2.set_ylabel("Fraction of data", color="orange")
    ax.set_title("Lift curve")


def plot_roc_curve(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true.tolist(), y_score.tolist())
    thr[0] = 1.0

    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, color="blue", marker="o", label="ROC curve")
    ax.grid()
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR/recall", color="blue")
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", linestyle="--")
    ax2 = ax.twinx()
    ax2.plot(fpr, thr, color="orange", marker="o", label="Score threshold")
    ax2.set_ylabel("Score threshold", color="orange")
    ax.set_title("ROC curve")


def analyze_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Statistics on data without using the model."""
    result = {}
    result["frac_pos"] = (
        data["onderzoekswaardig"].sum() / data["onderzoekswaardig"].count()
    )
    result["n_applications"] = data["onderzoekswaardig"].count()
    result["n_hh_onderzoek"] = data["is_onderzoek_hh"].sum()
    result["n_hh_screening"] = data["is_screening_hh"].sum()
    result["n_ic_screening"] = data["is_screening_ic"].sum()
    return result


def score(data: pd.DataFrame, model: SklearnPredictor) -> pd.DataFrame:
    preds = model.predict(data)
    scores = model.predict_proba(data)
    result = pd.DataFrame(
        np.column_stack((preds, scores[:, 1])),
        columns=["prediction", "score_onderzoekswaardig"],
    )
    return result


def calculate_model_performance(y_true, y_pred, y_score):
    support = (y_score > 0.5).sum() / y_score.count()
    baseline = y_true.sum() / y_true.count()
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    y_score = y_score.tolist()
    precision = precision_score(y_true, y_pred)
    lift = precision / baseline
    cf = confusion_matrix(y_true, y_pred, normalize="all")
    cfdf = pd.DataFrame(
        cf,
        columns=["prediction: niet ondzw", "prediction: wel ondzw"],
        index=["werkelijkheid: niet ondzw", "werkelijkheid: wel ondzw"],
    ).applymap(lambda v: f"{v:.1%}")

    result = {
        "precision": precision,
        "auc": roc_auc_score(y_true, y_score),
        "auc2": roc_auc_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "support": support,
        "lift": lift,
        "confusion_matrix": cfdf.to_markdown(),
    }
    print(result)
    return result


def calculate_lift_curve(
    y_true, y_score, baseline: float, score_levels: List[float] = None
) -> pd.DataFrame:
    if score_levels is None:
        score_levels = list(sorted(y_score.unique()))
        if score_levels[0] != 0:
            score_levels = [0.0] + score_levels
    lc = []
    for level in score_levels:
        pred = y_score >= level
        precision = precision_score(y_true.tolist(), pred.tolist())
        pct_data = pred.sum() / pred.count()
        lift = precision / baseline if pct_data else np.nan
        lc.append(
            {
                "precision": precision,
                "confidence": level,
                "lift": lift,
                "pct_data": pct_data,
            }
        )
    result = pd.DataFrame(lc)
    return result


def load_model(model_file: str) -> Dict[str, Any]:
    if model_file is None:
        with open_binary(
            "wpi_onderzoekswaardigheid_aanvraag.resources", "model.pkl"
        ) as f:
            model_dict = joblib.load(f)
    else:
        model_dict = joblib.load(model_file)
    return model_dict


def load_data(data_file: str) -> pd.DataFrame:
    return pd.read_pickle(data_file)


def plot_model_feature_importance(model_dict, n_feats=20):
    fimp_df = model_dict["feature_importance"]
    fimp_df = fimp_df.set_index("f_name")

    fig, ax = plt.subplots(1, 1)
    fimp_df.sort_values("f_imp", ascending=False).head(n_feats)["f_imp"].plot.bar()
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Feature importance")
    plt.xlabel("Feature name")
    plt.axhline(0, color="black")
    plt.title(f"Relative importance of {n_feats} most important features")
    plt.tight_layout()
    plt.grid()


def create_output_dir(output_dir: str) -> None:
    """Create output directory if it does not exist."""
    p = Path(output_dir)
    if p.exists():
        if not p.is_dir():
            raise ValueError(
                f'"{output_dir}" already exists, but it\'s not a directory.'
            )
    else:
        logger.info(f'Creating output directory for model statistics "{output_dir}".')
        p.mkdir(parents=True)


def _generate_cut_fimp_statement(
    cum_fimp_threshold, n_features, n_original_features=None
):
    if n_original_features is not None:
        return f"""Features above {cum_fimp_threshold} cumulative feature importance were cut. {n_features}/{n_original_features} were left."""
    else:
        return f"""Features above {cum_fimp_threshold} cumulative feature importance were cut. {n_features} were left."""


if __name__ == "__main__":
    main()


# TODO: Remove duplicate definition of `filter_application_handling`
