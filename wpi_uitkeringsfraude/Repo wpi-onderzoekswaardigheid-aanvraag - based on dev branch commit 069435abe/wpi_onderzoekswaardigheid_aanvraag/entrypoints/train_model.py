import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pretty_errors  # noqa: F401
from azureml.core import Run
from bias_collection.bias_analyzer import BiasAnalyzer
from bias_collection.reweigh_transformer import ReweighTransformer
from fraude_preventie import setup_logging
from fraude_preventie.evaluation.evaluation import (
    calc_bootstrapped_performance,
    evaluate_model,
)
from fraude_preventie.evaluation.feature_importance import (
    get_and_print_feature_importances,
)
from fraude_preventie.model.model_utils import add_mapped_feature_names_to_model
from fraude_preventie.preprocessing.feature_selection import ForwardFeatureSelection
from fraude_preventie.supertimer import timer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from wpi_onderzoekswaardigheid_aanvraag.entrypoints.model_stats import (
    generate_statistics_markdown,
)
from wpi_onderzoekswaardigheid_aanvraag.model.build_model import (
    filter_application_handling,
    make_dataframe_mapper,
    make_model,
    split_data_train_test,
)
from wpi_onderzoekswaardigheid_aanvraag.model.manage_model_info import (
    load_feature_list,
    load_parameter_settings,
    load_selected_features,
    save_params,
    save_selected_features,
)
from wpi_onderzoekswaardigheid_aanvraag.project_paths import ARTIFACT_PATH, INFO_PATH
from wpi_onderzoekswaardigheid_aanvraag.settings.flags import PipelineFlag
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import WPISettings

logger = logging.getLogger(__name__)

# Prevent dataframe mapper from printing time for each column transformation.
logging.getLogger("sklearn_pandas").setLevel(logging.WARNING)
# Prevent EBM from printing the same warning many times during forward feature selection.
logging.getLogger("interpret.utils.all").setLevel(logging.ERROR)


def train_and_store_model() -> Dict[str, Any]:
    """Trains and stores a model on data that has been transformed with the master
    pipeline.

    Returns
    -------
    result:
        dictionary with keys "model", "performance" and "feature_importance".
        This same dictionary will also be pickled to ``resources_folder``
    """
    label = "onderzoekswaardig"
    logger.info("Reading training data...")
    run = Run.get_context()
    transformed_data = run.input_datasets["transformed_data"].to_pandas_dataframe()

    include_handling_types = WPISettings.get_settings()["model"]["handling_types"]
    transformed_data = filter_application_handling(
        transformed_data, include_handling_types
    )

    logger.info(f"Number of entries: {len(transformed_data)}")
    logger.info(
        f"Percentage positives: "
        f"{round((transformed_data.onderzoekswaardig.isin([1, 2]).sum() * 100) / len(transformed_data.onderzoekswaardig), 1)}%"
    )

    model_dict, data_dict = train_and_test_model(transformed_data, label)
    model_dict[
        "model"
    ].handling_types = include_handling_types  # Save with model which handling types it was trained on.

    # Save model.
    mounted_output_dir = run.output_datasets["model"]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
    file_path = f"{mounted_output_dir}/wpi_model.pkl"
    logger.info(f"Compressing and writing trained model dictionary to {file_path}...")
    joblib.dump(value=model_dict, filename=file_path)
    run.upload_file("outputs/wpi_model.pkl", file_path)
    logger.info("Done.")

    # Also save feature importances separately for easy access.
    mounted_output_dir = run.output_datasets["feature_importances"]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
    file_path = f"{mounted_output_dir}/feature_importances.csv"
    logger.info(f"Writing feature importances to {file_path}...")
    model_dict["feature_importance"].sort_values(by="f_imp", ascending=False).to_csv(
        file_path, index=False
    )
    run.upload_file("outputs/feature_importances.csv", file_path)
    logger.info("Done.")

    flags = WPISettings.get_settings()["flags"]
    if flags & PipelineFlag.SAVE_PARAMS:
        mounted_output_dir = run.output_datasets["parameters"]
        os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
        file_path = f"{mounted_output_dir}/parameters.yml"
        save_params(
            model_dict["model"].named_steps["clf"].best_params_,
            file_path,
            algorithm_type=WPISettings.get_settings()["model"]["algorithm"],
        )
        run.upload_file("outputs/parameters.yml", file_path)

    generate_statistics_markdown(  # type: ignore  # Mypy cannot handle unpacking dicts
        **data_dict,
        model_dict=model_dict,
    )
    run.upload_folder("outputs/statistics", run.output_datasets["statistics"])
    run.upload_folder("outputs/pipeline", run.input_datasets["pipeline_file"])

    if WPISettings.get_settings()["model"]["register_model"]:
        run.register_model(
            model_name="wpi_model",
            model_path="outputs",
            tags={},
            datasets=[("transformed_data", run.input_datasets["transformed_data"])],
        )

    return model_dict


def train_and_test_model(
    transformed_data,
    label: str = "onderzoekswaardig",
):
    """Trains and tests the model on data that has been transformed with the master
    pipeline.

    Parameters
    ----------
    transformed_data:
        transformed data (output of the :class:`.MasterPipeline`)
    params_outpath:
        path where features above `cum_fimp_threshold` will be stored
    label:
        column name containing the label

    Returns
    -------
    result:
        dictionary with keys "model", "performance" and "feature_importance".
    """
    flags = WPISettings.get_settings()["flags"]
    do_gridsearch = flags & PipelineFlag.GRIDSEARCH
    do_reweigh = flags & PipelineFlag.REWEIGH

    feature_selection_method = WPISettings.get_settings()["model"]["feature_selection"]
    logger.info(f"Feature selection method is: {feature_selection_method}")

    do_forward_feature_selection = (
        feature_selection_method == "forward_feature_selection"
    )
    loaded_features = None
    selected_features = None
    cum_fimp_threshold = None

    if feature_selection_method == "selected_features":
        loaded_features = load_selected_features()

    if feature_selection_method in [
        "all_features",
        "selected_features",
        "forward_feature_selection",
    ]:
        model_dict, data_dict, selected_features = _train_and_test_model_on_cols(
            transformed_data,
            label,
            do_gridsearch,
            do_reweigh,
            do_forward_feature_selection=do_forward_feature_selection,
            cols_to_use=loaded_features,
        )

    elif feature_selection_method == "cut_fimp":
        cum_fimp_threshold = WPISettings.get_settings()["model"]["fimp_threshold"]
        if cum_fimp_threshold is None:
            logger.error(
                "Config must contain entry `fimp_threshold` when feature selection method is `cut_fimp`"
            )

        logger.info("Training model on all features")
        model_dict, data_dict, _ = _train_and_test_model_on_cols(
            transformed_data, label, do_gridsearch
        )

        original_auc = model_dict["performance"]["auc"]
        logger.info(f"AUC before trimming features: {original_auc:.3f}")

        n_original_features = len(model_dict["model"].raw_feature_names)

        # After training with all the features, we determine which columns should
        # be included at the given feature importance threshold value.
        threshold_index = _calc_threshold_index(
            model_dict["feature_importance"], cum_fimp_threshold
        )
        selected_features = (
            model_dict["feature_importance"]["f_name"].head(threshold_index).to_numpy()
        )

        logger.info(
            f"Retraining model with only the most important features "
            f"contributing to the cumulative feature importance threshold: "
            f"{cum_fimp_threshold}"
        )

        # If cutting features, don't do another gridsearch, just retrain with
        # fewer features.
        logger.info("Turning off gridsearch for retraining on trimmed features")
        logger.info(
            f"There are {len(selected_features)} features above the feature "
            f"importance threshold: {selected_features}"
        )

        model_dict, data_dict, _ = _train_and_test_model_on_cols(
            transformed_data,
            label,
            do_gridsearch=False,
            cols_to_use=selected_features,
        )

        logger.info(
            f"AUC after trimming features: {model_dict['performance']['auc']:.3f}"
        )

        model_dict["cum_fimp_threshold"] = cum_fimp_threshold
        model_dict["n_original_features"] = n_original_features

    else:
        logger.error(
            f"Invalid entry for config parameter `feature_selection`, allowed values are 'all_features', 'selected_features', 'cut_fimp', 'forward_feature_selection', input given was: {feature_selection_method}"
        )

    if feature_selection_method in ["forward_feature_selection", "cut_fimp"]:
        selected_features = (
            list(selected_features)
            if not isinstance(selected_features, list)
            else selected_features
        )
        info_to_save = {
            "selection_method": feature_selection_method,
            "features": selected_features,
        }

        if cum_fimp_threshold:
            info_to_save["fimp_threshold"] = cum_fimp_threshold

        save_selected_features(
            info_to_save,
            output_dataset_name="selected_features",
        )
        run = Run.get_context()
        run.upload_file(
            "outputs/selected_features.yml", run.output_datasets["selected_features"]
        )

    model_dict["flags"] = flags
    model_dict["feature_selection_method"] = feature_selection_method
    return model_dict, data_dict


def _train_and_test_model_on_cols(
    transformed_data,
    label: str = "onderzoekswaardig",
    do_gridsearch: Union[bool, PipelineFlag] = False,
    do_reweigh: Union[bool, PipelineFlag] = False,
    do_forward_feature_selection: Union[bool, PipelineFlag] = False,
    cols_to_use: List[str] = None,
):
    """Trains and tests the model on a specific set of columns in the transformed data.
    When the BIAS flag is active, a bias analysis will be carried out according to the
    specifications in the config file.

    Parameters
    ----------
    transformed_data:
        transformed data (output of the :class:`.MasterPipeline`)
    label:
        column name containing the label
    do_gridsearch:
        whether or not to carry out a gridsearch over the parameters
    cols_to_use:
        list of columns to use

    Returns
    -------
    result:
        dictionary with keys "model", "performance" and "feature_importance".
    """
    (
        cat_cols,
        num_cols,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = _prepare_train_test_data(transformed_data, label, only_use_columns=cols_to_use)

    steps = []

    # Add preprocessing to pipeline steps.
    mapper = make_dataframe_mapper(cat_cols, num_cols)
    steps.append(("prep", mapper))

    algorithm_type = WPISettings.get_settings()["model"]["algorithm"]

    if do_forward_feature_selection:
        # Model used to estimate feature set candidates
        params = load_parameter_settings(algorithm_type, load_grid=False)
        # Flatten dictionary first, because it's written in lists to work with gridsearch.
        params = {k: v[0] for k, v in params.items()}
        fs_model = make_model(algorithm_type, **params)

        fs = ForwardFeatureSelection(
            fs_model,
            cv=2,
            speculative_rounds=3,
            scoring="roc_auc",
            minimum_improvement=0.0005,
        )
        steps.append(("feature_selection", fs))

    # Add model cross-validation object to pipeline steps.
    clf = make_model(algorithm_type)
    cv_obj = _create_cv_object_from_model(
        clf,
        do_gridsearch=do_gridsearch,
        algorithm_type=algorithm_type,
    )
    steps.append(("clf", cv_obj))

    # Create pipeline from steps.
    pl = Pipeline(steps)

    if do_reweigh:
        # We calculate the sample weights outside of the pipeline instead of using the ReweighTransformer,
        # because else we run into problems with the dataframe mapper expecting certain sensitive (reweighing)
        # attributes to be there at predict time.
        logger.info("Calculating sample weights for reweighing")
        reweigh_features = WPISettings.get_settings()["reweigh_features"]
        rw = ReweighTransformer(
            reweigh_features_settings=reweigh_features, label_name="onderzoekswaardig"
        )
        rw.fit(X_train, y_train)
        sample_weights = rw.sample_weights_
    else:
        sample_weights = None

    # Optimize model parameters.
    with timer("Fitting and optimizing model"):
        pl.fit(
            X_train,
            y_train,
            **{"clf__sample_weight": sample_weights},
        )

    # Print results.
    logger.info(
        f"Tested {len(cv_obj.cv_results_['params'])} "
        f"candidate parameter settings. Using {cv_obj.best_params_}."
    )
    logger.info(f"Best performance: {cv_obj.best_score_:2f}")

    # Create feature list and add to model.
    prep_steps = pl.steps[:-1]
    mapped_df = X_train
    for s in prep_steps:
        mapped_df = s[1].transform(mapped_df)
    pl = add_mapped_feature_names_to_model(pl, cat_cols + num_cols, mapped_df)

    if WPISettings.get_settings()["flags"] & PipelineFlag.BIAS:
        data_to_analyze = X_test
        for s in prep_steps:
            data_to_analyze = s[1].transform(data_to_analyze)
        data_to_analyze["onderzoekswaardig"] = y_test.replace({True: 1, False: 0})
        BiasAnalyzer().analyze_features(
            data_to_analyze=data_to_analyze,
            model=pl.named_steps["clf"].best_estimator_,
            sensitive_features=WPISettings.get_settings()["sensitive_features"],
            outpath=Path(INFO_PATH) / "bias_report",
            thresholds=np.arange(0.4, 0.6, 0.01),
            label_column_name="onderzoekswaardig",
        )

    # Bootstrap performance on test set.
    n_iter = 10
    test_ratio = 0.8
    target_names = ["niet ondzw", "wel ondzw"]
    bootstrapped_performance = calc_bootstrapped_performance(
        X_test, y_test, pl, target_names, n_iter, test_ratio
    )
    logger.info(
        f"Average and std of performance across {n_iter} random samples "
        f"of {test_ratio * 100}% of the test data:\n"
        f"{pformat(bootstrapped_performance, width=120)}"
    )

    fimp = get_and_print_feature_importances(
        model=pl.named_steps["clf"].best_estimator_,
        mapped_feature_names=pl.mapped_feature_names,
        fimp_type="permutation",
        X_train_mapped=mapped_df,
        y_train=y_train,
    )

    test_performance = evaluate_model(pl, X_test, y_test)
    logger.info(
        f"Performance on full test data:\n{pformat(test_performance, width=120)}"
    )

    train_performance = evaluate_model(pl, X_train, y_train)

    model_dict = {
        "model": pl,
        "performance": test_performance,
        "bootstrapped_performance": bootstrapped_performance,
        "train_performance": train_performance,
        "feature_importance": fimp,
    }

    data_dict = {
        "train_data": X_train.assign(onderzoekswaardig=y_train),
        "test_data": X_test.assign(onderzoekswaardig=y_test),
    }

    if do_forward_feature_selection:
        selected_feature_names = pl.named_steps["feature_selection"].selected_extra_
    else:
        selected_feature_names = None

    return model_dict, data_dict, selected_feature_names


def _prepare_train_test_data(transformed_data, label, only_use_columns=None):
    """Splits the data into train and test, selecting only relevant columns.

    Parameters
    ----------
    transformed_data:
        transformed data (output of the :class:`.MasterPipeline`)
    label:
        column name containing the label
    only_use_columns:
        if it is not None, then only columns from this list will be used as features.

    Returns
    -------
    cat_cols:
        Categorical columns to be used for training.
    num_cols:
        Numerical columns to be used for training.
    X_train:
        Training set.
    y_train:
        Training label.
    X_test:
        Testing set.
    y_test:
        Testing label.
    """
    num_cols, cat_cols = load_feature_list()

    if only_use_columns is not None:
        cat_cols = [
            c
            for c in cat_cols
            for column_to_keep in only_use_columns
            if c in column_to_keep
        ]
        cat_cols = list(set(cat_cols))
        num_cols = [
            c
            for c in num_cols
            for column_to_keep in only_use_columns
            if c in column_to_keep
            # Check it with this if-statement, because a feature name from `num_cols` can be different from
            # the name in `only_use_columns`, due to imputation that may change the feature names.
        ]
        num_cols = list(set(num_cols))

        cols_to_use_without_num_cat_info = [
            f
            for f in list(set(only_use_columns) - set(cat_cols).union(set(num_cols)))
            if "imputed" not in f
        ]
        if len(cols_to_use_without_num_cat_info) > 0:
            logger.warning(
                f"Some columns are specified in `only_use_columns` that are not present in the "
                f"numerical/categorical feature information: {cols_to_use_without_num_cat_info}"
            )

    # Some columns we need to keep for evaluation, reweighing, and/or joining attributes in
    # the bias analysis.
    other_cols_to_keep = [
        label,
        "application_dienstnr",
        "is_onderzoek_hh",
        "is_screening_hh",
        "is_screening_ic",
        "subjectnr",
        "dtaanvraag",
    ]

    if WPISettings.get_settings()["flags"] & PipelineFlag.REWEIGH:
        reweigh_cols = list(WPISettings.get_settings()["reweigh_features"].keys())
        cols_not_present = set(reweigh_cols).difference(set(transformed_data.columns))
        if len(cols_not_present) > 0:
            raise RuntimeError(
                f"Some columns required for reweighing are not present in the input dataframe: {cols_not_present}"
            )
    else:
        reweigh_cols = []

    all_columns = set(transformed_data.columns)
    transformed_data_filtered = transformed_data[
        cat_cols + num_cols + other_cols_to_keep + reweigh_cols
    ]

    unused_columns = all_columns - set(num_cols + cat_cols + other_cols_to_keep)
    logger.info(f"Training on columns {list(num_cols + cat_cols)}")
    if len(unused_columns) > 0:
        logger.debug(
            f"The following columns are part of the data, "
            f"but will not be used to train the model: {unused_columns}"
        )
    logger.debug("Splitting data...")

    X_train, X_test, y_train, y_test = split_data_train_test(
        transformed_data_filtered, shuffle=True, label=label, random_state=42
    )
    with open(ARTIFACT_PATH / "test_set_ids.txt", "w") as f:
        f.writelines(",".join(map(str, X_test["application_dienstnr"])))

    return cat_cols, num_cols, X_train, y_train, X_test, y_test


def _create_cv_object_from_model(
    model, do_gridsearch: Union[bool, PipelineFlag], algorithm_type: str
):
    param_grid = load_parameter_settings(algorithm_type, load_grid=do_gridsearch)

    if do_gridsearch:
        logger.info(f"Doing gridsearch over: {param_grid}.")
    else:
        logger.info(f"Skipping gridsearch. Using parameters: {param_grid}.")

    cv_obj = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=4,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=2,
        error_score="raise",
        return_train_score=True,
    )

    return cv_obj


def _calc_threshold_index(df, threshold_value):
    """Sort the importance values and find the index of the feature that matches
    the cumulative importance threshold."""
    df = df[df["f_imp"] >= 0]
    df = df.sort_values("f_imp", ascending=False)
    # Normalize the feature importances to add up to one
    df["normalized_importance"] = df["f_imp"] / df["f_imp"].sum()
    df["cumulative_importance"] = np.cumsum(df["normalized_importance"])
    return np.min(np.nonzero(df["cumulative_importance"].values >= threshold_value))


def main():
    run = Run.get_context()
    config_file = "dev-config.yml"
    settings = WPISettings.set_from_yaml(config_file)
    setup_logging(settings["logging"])
    logger.info(f"Active pipeline flags: {settings['flags']}")
    train_and_store_model()
    run.complete()


if __name__ == "__main__":
    main()
