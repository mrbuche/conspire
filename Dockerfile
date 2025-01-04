FROM julia:1.11
FROM python:3.13
FROM rust:1.83.0
ARG VERSION
RUN if [[ -z "$VERSION" ]]; \
    then pip install conspire==$VERSION; \
    else pip install pip@git+https://github.com/mrbuche/conspire.py; \
    fi

