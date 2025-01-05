FROM python:3.13
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN pip install pip@git+https://github.com/mrbuche/conspire.py
