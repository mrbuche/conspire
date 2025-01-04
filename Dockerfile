FROM julia:1.11 as julia
FROM python:3.13
ARG VERSION
RUN if [[ -z "$VERSION" ]]; \
    then pip install conspire==$VERSION; \
    else pip install pip@git+https://github.com/mrbuche/conspire.py; \
    fi

