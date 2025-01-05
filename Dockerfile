FROM python:latest
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install git+https://github.com/mrbuche/conspire.py.git
