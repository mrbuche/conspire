FROM rust:1.83.0 AS rust

RUN which cargo

FROM python:3.13

RUN pip install pip@git+https://github.com/mrbuche/conspire.py
