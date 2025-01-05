FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.juliaup/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        julia -e 'using Pkg; Pkg.add(url="https://github.com/mrbuche/Conspire.jl")'; \
        git clone https://github.com/mrbuche/conspire.py.git; \
    else \
        pip install conspire==$VERSION pytest; \
        julia -e 'using Pkg; Pkg.add(name="Conspire", version="'$VERSION'")'; \
        git clone https://github.com/mrbuche/conspire.py.git --branch $VERSION; \
    fi
RUN mv conspire.py/pyproject.toml .
RUN mv conspire.py/tests/ .
RUN rm -r conspire.py/
