# syntax=docker/dockerfile:experimental

FROM python:3.6.8-slim-stretch as builder

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get -y install --no-install-recommends \
        openssh-client \
    && apt-get autoremove --purge \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip-tools --no-cache-dir

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY docker/requirements.in /opt/requirements.in

RUN --mount=type=ssh mkdir -p -m 0600 ~/.ssh \
    && ssh-keyscan -p 7999 git.ba.innovatrics.net >> ~/.ssh/known_hosts \
    && pip-compile /opt/requirements.in
