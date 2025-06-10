#!/bin/bash
export DEFAULT_CONNECTION="azure_api"
service ssh start

uvicorn wpi_onderzoekswaardigheid_aanvraag.api.api:endpoint --host=0.0.0.0 --port=8000 --workers 4 --limit-concurrency 10

tail -F /dev/null
