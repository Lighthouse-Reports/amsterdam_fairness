import argparse
import json
import logging
import os
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Dict, List, Union

import fastapi
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from wpi_onderzoekswaardigheid_aanvraag.api.healthcheck import WPIHealthcheck
from wpi_onderzoekswaardigheid_aanvraag.scorer import Scorer
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import (
    WPISettings,
    setup_azure_logging,
)

logger = logging.getLogger(__name__)
scorer: Scorer = None  # will be initialized in setup()
endpoint = FastAPI(default_response_class=fastapi.responses.ORJSONResponse)

SUCCESS = 200
FAILURE = 500

# Set defaults for arguments
classification_threshold = 0.63
model_file = "model_artifacts/model.pkl"
pipeline_file = "model_artifacts/pipeline.pkl"
config_file_path = "production-config.yml"


class ApplicationsScoreModel(BaseModel):
    """Model of how application score requests should be structured."""

    application_dienstnr: List[Union[str, int]]
    auth_token: str


class ApplicationScoreModel(BaseModel):
    """Model of how application score requests should be structured."""

    application_dienstnr: Union[str, int]
    auth_token: str


class SuccessfulPrediction(BaseModel):
    """Model of a single, succesful prediction returned by the API."""

    application_dienstnr: Union[int, str]
    subjectnr: int
    results_type: int
    prediction: int
    score: float
    feature_contributions: Dict[str, float]
    feature_values: Dict[str, float]
    has_ams_address: bool
    warnings: str


class FailedPrediction(BaseModel):
    """Model of a single, failed prediction returned by the API."""

    application_dienstnr: Union[int, str]
    error: str


class ScoreResponse(BaseModel):
    """Model of JSON returned by the API."""

    status_code: int
    timestamp: datetime
    n_results: int
    n_errors: int
    results: List[SuccessfulPrediction]
    errors: List[FailedPrediction]
    aml_model_id: str
    project_version: str


def authenticate_and_log_request(post_function):
    @wraps(post_function)
    async def wrapper_params(params, request):
        if len(params.auth_token) != len(settings["api"]["auth_token"]):
            logger.warning(
                f"Received {len(params.auth_token)} characters token "
                f'but expected {len(settings["api"]["auth_token"])} characters token'
            )
        if params.auth_token == settings["api"]["auth_token"]:
            logger.info(
                f"Request URL: {request.url}\n"
                f"Request method: {request.method}\n"
                f"Application requested: {params.application_dienstnr}\n"
                f"Request headers: {request.headers}\n"
                f"Request path params: {request.path_params}\n"
                f"Request query params: {request.query_params}\n"
                f"Request cookies: {request.cookies}\n"
                f"Request headers x-original-host: {request.headers.get('x-original-host', None)}\n"
                f"Request headers x-client-ip:{request.headers.get('x-client-ip', None)}\n"
                f"Request headers X-Forwarded-For: {request.headers.get('X-Forwarded-For', None)}\n"
            )
            json_result = await post_function(params, request)
            logger.info(
                f"Response status code:{json_result['status_code']}\n"
                f"Response timestamp:{json_result['timestamp']}\n"
                f"Response n_results:{json_result['n_results']}\n"
                f"Response n_errors:{json_result['n_errors']}\n"
                f"Response aml_model_id:{json_result['aml_model_id']}\n"
                f"Response project_version:{json_result['project_version']}\n"
                f"Request URL: {request.url}\n"
                f"Request method: {request.method}\n"
                f"Applications requested: {params.application_dienstnr}\n"
                f"Request headers: {request.headers}\n"
                f"Request path params: {request.path_params}\n"
                f"Request query params: {request.query_params}\n"
                f"Request cookies: {request.cookies}\n"
                f"Request headers x-original-host: {request.headers.get('x-original-host', None)}\n"
                f"Request headers x-client-ip:{request.headers.get('x-client-ip', None)}\n"
                f"Request headers X-Forwarded-For: {request.headers.get('X-Forwarded-For', None)}\n"
            )
            if json_result["status_code"] == 200:
                for result in json_result["results"]:
                    logger.info(f"Full response: {json.dumps(result.to_json())}")
            return json_result
        else:
            raise HTTPException(status_code=401, detail="Authentication failed.")

    return wrapper_params


@endpoint.post("/test", status_code=SUCCESS)
def test_endpoint():
    return {"response": "Endpoint works correctly."}


@endpoint.get("/healthcheck", status_code=SUCCESS)
async def healthcheck():
    result = await WPIHealthcheck(scorer).run_healthchecks()
    logger.info(result)

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=FAILURE, detail=result)


@endpoint.post(
    "/score_application",
    response_model=ScoreResponse,
    status_code=SUCCESS,
)
@authenticate_and_log_request
async def score_application(params: ApplicationScoreModel, request: Request):
    """API endpoint that creates predictions for the specified application.
    The application_dienstnr must be valid dienstnr from the Socrates Dienst table.

    Example
    ----------
    url: url of the endpoint.
    data: Json containing the following data:
        - Dienstnr of application to score
        - Authorization token

    requests.post(url, json=data)

    Parameters
    ----------
    params:
        Application to score.

    request:
        Entire request object.

    Returns
    -------
    predictions:
        The output is a json dictionary with results and errors.
        Results are a list of dictionaries with the following keys:
        - application_dienstnr: Dienstnr of the application that was scored.
        - subjectnr: Subject number belonging to the application. Also called administratienummer
          or adminnummer in some systems.
        - results_type: Integer with meaning 1 - Onderzoekswaardig; 2 - Niet onderzoekswaardig, 3 - IOAW/IOAZ;
          4 - Missing prediction other reasons
        - prediction: 1 if application is predicted to be 'onderzoekswaardig', i.e.
          if the score for the application is at least as high as the classification
          threshold; 0 if not.
        - score: Risk score of the application.
        - feature_contributions: Dictionary of the contributions of each feature to the final score,
          structured as {feature: contrib}
        - feature_values: Dictionary of the feature values, structured as {feature: value}
        - has_ams_address: Boolean indicating whether the applicant has a (known) address in Amsterdam
        - warnings: String with warnings thrown during the scoring
    """
    results_list = []
    errors_list = []
    try:
        result = await scorer.score(
            tuple([params.application_dienstnr]), config_file_path=config_file_path
        )
        results_list.append(result.reset_index().squeeze())
    except Exception:
        errors_list.append(
            {
                "application_dienstnr": params.application_dienstnr,
                "error": traceback.format_exc(),
            }
        )

    for i in range(len(results_list)):
        results_list[i] = clean_float32(results_list[i])

    json_result = {
        "status_code": SUCCESS,
        "timestamp": int(time.time()),
        "n_results": len(results_list),
        "n_errors": len(errors_list),
        "results": results_list,
        "errors": errors_list,
        "aml_model_id": os.getenv("AML_MODEL_ID"),
        "project_version": os.getenv("PROJECT_VERSION"),
    }
    return json_result


def clean_float32(result_dict: Dict):
    for key, value in result_dict.items():
        if isinstance(value, Dict):
            result_dict[key] = clean_float32(value)
        elif isinstance(value, np.float32):
            result_dict[key] = float(value)
    return result_dict


def setup(
    config_file,
    classification_threshold,
    pipeline_file,
    model_file,
):
    WPISettings.setup_production_settings(config_file=config_file)
    setup_azure_logging(WPISettings.get_settings()["logging"])

    # Initialize scorer globally, this way the scorer is initialized only once.
    global scorer
    scorer = Scorer(
        classification_threshold=classification_threshold,
        pipeline_file=pipeline_file,
        model_file=model_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create predictions for new cases with a pretrained model. "
        "Needs to output of train_pipeline, train_model and a data folder."
    )
    parser.add_argument(
        "--pipeline_file",
        help="Path to pickle file of fitted pipeline. If omitted, then the pipeline shipped with the package is used.",
        required=False,
        default=pipeline_file,
    )
    parser.add_argument(
        "--model_file",
        help="Path to pickle file of fitted model. If omitted, then the model shipped with the package is used",
        required=False,
        default=model_file,
    )
    parser.add_argument(
        "--classification_threshold",
        help="Minimum score required to give a positive prediction. Should be between 0 and 1. Defaults to 0.5.",
        required=False,
        default=classification_threshold,
    )
    parser.add_argument(
        "--config",
        default=config_file_path,
        help="Path to the config file.",
        required=False,
    )

    args = parser.parse_args()

    # This is ugly, but needed to pass the config_file_path to Scorer.score()
    config_file_path = args.config

    setup(
        config_file_path,
        float(args.classification_threshold),
        args.pipeline_file,
        args.model_file,
    )
    settings = WPISettings.get_settings()

    logger.info(f'AML model ID: {os.getenv("AML_MODEL_ID")}')
    logger.info(f'Project version: {os.getenv("PROJECT_VERSION")}')

    # Run API.
    uvicorn.run(
        endpoint,
        host=settings["api"]["host"],
        port=settings["api"]["port"],
        log_level="debug",
    )

else:
    setup(
        config_file_path,
        float(classification_threshold),
        pipeline_file,
        model_file,
    )
    settings = WPISettings.get_settings()

    logger.info(f'AML model ID: {os.getenv("AML_MODEL_ID")}')
    logger.info(f'Project version: {os.getenv("PROJECT_VERSION")}')
