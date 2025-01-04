# FROM julia:1.11 as julia
FROM python:3.13
# FROM rust:1.83.0
RUN pip install pip@git+https://github.com/mrbuche/conspire.py
