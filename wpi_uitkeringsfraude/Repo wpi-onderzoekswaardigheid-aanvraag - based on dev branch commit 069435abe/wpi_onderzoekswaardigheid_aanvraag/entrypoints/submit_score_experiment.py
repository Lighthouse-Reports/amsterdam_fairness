import pkg_resources
import yaml
from amla_toolkit import AMLInterface
from azureml.core import Dataset, Datastore, Environment
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

    env = Environment.get(
        workspace=aml_interface.workspace, name=experiment_details["env_name"]
    )
    ppl_config = RunConfiguration()
    ppl_config.environment = env

    application_dienstnr_to_score = Dataset.get_by_name(
        aml_interface.workspace, "application_dienstnr_to_score"
    )

    scorer_results = OutputFileDatasetConfig(
        name="scorer_results",
        destination=(datastore, "/model_scoring/{output-name}"),
    ).as_upload(overwrite=True)

    score_applications = PythonScriptStep(
        name="score_applications",
        script_name="wpi_onderzoekswaardigheid_aanvraag/entrypoints/score_applications.py",
        inputs=[
            application_dienstnr_to_score.as_named_input(
                "application_dienstnr_to_score"
            )
        ],
        outputs=[scorer_results],
        compute_target=aml_interface.get_compute_target(
            experiment_details["compute_name"]
        ),
        source_directory=experiment_details["src_dir"],
        runconfig=ppl_config,
    )

    run = aml_interface.create_pipeline([score_applications], "wpi-score-pipeline")
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
