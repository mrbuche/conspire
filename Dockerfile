FROM python:3.13
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -y
RUN pip install pip@git+https://github.com/mrbuche/conspire.py
