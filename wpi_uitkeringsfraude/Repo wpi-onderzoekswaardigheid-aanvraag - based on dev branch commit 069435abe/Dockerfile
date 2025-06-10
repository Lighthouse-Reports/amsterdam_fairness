FROM python:3.8.12-slim as builder

WORKDIR /usr/src

# Retrieve credentials to Azure Artifacts from the Docker build args.
ARG POETRY_HTTP_BASIC_GEMEENTE_PASSWORD_arg
ENV POETRY_HTTP_BASIC_GEMEENTE_PASSWORD=$POETRY_HTTP_BASIC_GEMEENTE_PASSWORD_arg

# Username is irrelevant, so just set it here.
ENV POETRY_HTTP_BASIC_GEMEENTE_USER=docker_wpi

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3013
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends -y gcc=4:10.2.1-1 curl=7.74.0-1.3+deb11u5 build-essential=12.9 \
    libpq-dev=13.9-0+deb11u1 python2-dev=2.7.18-3 libfreetype6-dev=2.10.4+dfsg-1+deb11u1 pkg-config=0.29.2-1 \
 && pip install --no-cache-dir --upgrade pip \
 && curl -sSL https://install.python-poetry.org | python3 - --version 1.1.15 \
 && /root/.local/bin/poetry config virtualenvs.create false

COPY wpi-onderzoekswaardigheid-aanvraag/pyproject.toml .
COPY wpi-onderzoekswaardigheid-aanvraag/poetry.lock .

ENV PATH="/opt/venv/bin:$PATH"

RUN python -mvenv /opt/venv \
 && . /opt/venv/bin/activate \
 && /root/.local/bin/poetry install --no-ansi --no-interaction --no-dev --no-root

# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -I wheel>=0.38.1 \
 && pip install --no-cache-dir -I pyOpenSSL==22.1.

FROM python:3.8.12-slim

WORKDIR /usr/src

RUN apt-get update \
 && apt-get upgrade -y --fix-missing \
 && echo "Europe/Amsterdam" > /etc/timezone \
 && rm /etc/localtime \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends -y libpq-dev=13.9-0+deb11u1 \
 && dpkg-reconfigure -f noninteractive tzdata \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/src/poetry.lock .

ENV PATH="/opt/venv/bin:$PATH"

COPY wpi-onderzoekswaardigheid-aanvraag/production-config.yml .
COPY wpi-onderzoekswaardigheid-aanvraag/entrypoint.sh .
COPY wpi-onderzoekswaardigheid-aanvraag/pyproject.toml .
COPY model_artifacts/wpi_model/outputs/wpi_model.pkl model_artifacts/model.pkl
COPY model_artifacts/wpi_model/outputs/pipeline/pipeline.pkl model_artifacts/pipeline.pkl
COPY wpi-onderzoekswaardigheid-aanvraag/wpi_onderzoekswaardigheid_aanvraag wpi_onderzoekswaardigheid_aanvraag

COPY wpi-onderzoekswaardigheid-aanvraag/sshd_config /etc/ssh/
RUN apt-get update \
 && apt-get install -y --no-install-recommends dialog \
 && apt-get install -y --no-install-recommends htop \
 && UCF_FORCE_CONFFOLD=1 apt-get install -y --no-install-recommends openssh-server \
 && echo "root:Docker!" | chpasswd \
 && chmod u+x ./entrypoint.sh
EXPOSE 2222

# Overwrite password for security reasons
ENV POETRY_HTTP_BASIC_GEMEENTE_PASSWORD=bogus

# Make sure to do this in the second half of the Dockerfile, else it doesn't get included, since we only copy the venv.
ARG AML_MODEL_ID_arg
ENV AML_MODEL_ID=$AML_MODEL_ID_arg
ARG PROJECT_VERSION_arg
ENV PROJECT_VERSION=$PROJECT_VERSION_arg

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
