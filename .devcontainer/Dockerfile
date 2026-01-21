# syntax=docker/dockerfile:1.7-labs

FROM mcr.microsoft.com/devcontainers/python:3.14-trixie

ARG USERNAME="vscode"

ENV VIRTUAL_ENV="/tmp/venv"
ENV HF_HOME="/hf/"

ENV UV_PROJECT_ENVIRONMENT="${VIRTUAL_ENV}"
ENV UV_LINK_MODE="copy"

RUN mkdir -p "${HF_HOME}" \
    && chmod 777 "${HF_HOME}"

USER "${USERNAME}"

RUN mkdir -p "${VIRTUAL_ENV}"

COPY --from=ghcr.io/astral-sh/uv:latest \
    /uv /uvx \
    /bin/

RUN cat <<EOF >> ~/.bashrc
if [[ -d "${VIRTUAL_ENV}/bin/" ]]; then
    source "${VIRTUAL_ENV}/bin/activate"
fi
EOF

VOLUME "${VIRTUAL_ENV}"
VOLUME "${HF_HOME}"
