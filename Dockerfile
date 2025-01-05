FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | bash -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install git+https://github.com/mrbuche/conspire.py.git
