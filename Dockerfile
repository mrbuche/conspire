FROM julia:1.11 as julia
FROM python:3.13
RUN pip install conspire==0.1.1
