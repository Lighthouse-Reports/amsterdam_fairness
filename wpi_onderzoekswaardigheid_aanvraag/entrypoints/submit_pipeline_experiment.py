import pkg_resources
import yaml
from amla_toolkit import AMLInterface
from azureml.core import Datastore, Environment
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep


def main():
    aml_interface = AMLInterface("aml-config.yml", "analyse_services", dtap="ont")

    with open("aml-config.yml", "r") as fs:
        aml_config = yaml.safe_load(fs)
        experiment_details = aml_config["experiment_details"]

    packages_and_versions_local_env = dict(
        tuple(str(ws).split()) for ws in pkg_resources.working_set
    )
    packages_and_versions_local_env["setuptools"] = "59.8.0"
    packages_and_versions_local_env.pop("wpi-onderzoekswaardigheid-aanvraag", None)
    packages = [
        f"{package}=={version}"
        for package, version in packages_and_versions_local_env.items()
    ]

    pat_token = experiment_details["pat_token"]
    aml_interface.workspace.set_connection(
        name="connection-daso",
        category="PythonFeed",
        target="https://pkgs.dev.azure.com/CloudCompetenceCenter/Datateam-Sociaal",
        authType="PAT",
        value=pat_token,
    )
    aml_interface.create_aml_environment(
        experiment_details["env_name"],
        base_dockerfile="DockerfileAML",
        pip_packages=packages,
        pip_option="--extra-index-url https://pkgs.dev.azure.com/CloudCompetenceCenter/Datateam-Sociaal/_packaging/team-AA/pypi/simple/",
    )

    datastore = Datastore(aml_interface.workspace, experiment_details["datastore_name"])

    transformed_data = (
        OutputFileDatasetConfig(
            name="transformed_data",
            destination=(datastore, "/preprocessing/{output-name}"),
        )
        .as_upload(overwrite=True)
        .read_parquet_files()  # To promote File to Tabular Dataset
        .register_on_complete(name="transformed_data")
    )
    pipeline_file = OutputFileDatasetConfig(
        name="pipeline_file",
        destination=(datastore, "/preprocessing/{output-name}"),
    ).as_upload(overwrite=True)

    env = Environment.get(
        workspace=aml_interface.workspace, name=experiment_details["env_name"]
    )
    ppl_config = RunConfiguration()
    ppl_config.environment = env

    train_preprocessing_pipeline = PythonScriptStep(
        name="train_preprocessing_pipeline",
        script_name="wpi_onderzoekswaardigheid_aanvraag/entrypoints/train_preprocessing_pipeline.py",
        outputs=[transformed_data, pipeline_file],
        compute_target=aml_interface.get_compute_target(
            experiment_details["compute_name"]
        ),
        source_directory=experiment_details["src_dir"],
        allow_reuse=True,
        runconfig=ppl_config,
    )

    selected_features = OutputFileDatasetConfig(
        name="selected_features",
        destination=(datastore, "/model_training/{output-name}"),
    ).as_upload(overwrite=True)

    model = OutputFileDatasetConfig(
        name="model",
        destination=(datastore, "/model_training/{output-name}"),
    ).as_upload(overwrite=True)

    feature_importances = OutputFileDatasetConfig(
        name="feature_importances",
        destination=(datastore, "/model_training/{output-name}"),
    ).as_upload(overwrite=True)

    parameters = OutputFileDatasetConfig(
        name="parameters",
        destination=(datastore, "/model_training/{output-name}"),
    ).as_upload(overwrite=True)

    statistics = OutputFileDatasetConfig(
        name="statistics",
        destination=(datastore, "/model_training/info/model_stats/{output-name}"),
    ).as_upload(overwrite=True)

    train_model = PythonScriptStep(
        name="train_model",
        script_name="wpi_onderzoekswaardigheid_aanvraag/entrypoints/train_model.py",
        inputs=[
            transformed_data.as_input(name="transformed_data"),
            pipeline_file.as_input(name="pipeline_file").as_download(),
        ],
        outputs=[selected_features, model, feature_importances, parameters, statistics],
        compute_target=aml_interface.get_compute_target(
            experiment_details["compute_name"]
        ),
        source_directory=experiment_details["src_dir"],
        runconfig=ppl_config,
    )
    run = aml_interface.create_pipeline(
        [train_preprocessing_pipeline, train_model], "wpi-pipeline"
    )
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
