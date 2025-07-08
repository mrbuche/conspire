FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.juliaup/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        git clone https://github.com/mrbuche/conspire.py.git; \
        git clone https://github.com/mrbuche/Conspire.jl.git; \
        julia -e 'using Pkg; Pkg.add(path="Conspire.jl")'; \
    else \
        pip install conspire==$VERSION pytest; \
        git clone https://github.com/mrbuche/conspire.py.git --branch $VERSION; \
        git clone https://github.com/mrbuche/Conspire.jl.git --branch $VERSION; \
        julia -e 'using Pkg; Pkg.add(path="Conspire.jl")'; \
    fi
RUN mkdir -p /usr/conspire/python/
RUN mv conspire.py/pyproject.toml /usr/conspire/python/
RUN mv conspire.py/tests/ /usr/conspire/python/
RUN rm -r conspire.py/
RUN rm -r Conspire.jl/
