FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -s -- -y
RUN curl https://astral.sh/uv/install.sh -fsSL | sh
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.juliaup/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        uv pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        git clone https://github.com/mrbuche/conspire.py.git; \
        git clone https://github.com/mrbuche/Conspire.jl.git --recurse-submodules; \
        julia -e 'using Pkg; Pkg.develop(path="Conspire.jl"); Pkg.build("Conspire")'; \
    else \
        uv pip install conspire==$VERSION pytest; \
        git clone https://github.com/mrbuche/conspire.py.git --branch $VERSION; \
        git clone https://github.com/mrbuche/Conspire.jl.git --branch $VERSION --recurse-submodules; \
        julia -e 'using Pkg; Pkg.develop(path="Conspire.jl"); Pkg.build("Conspire")'; \
    fi
RUN mkdir -p /usr/conspire/python/
RUN mv conspire.py/pyproject.toml /usr/conspire/python/
RUN mv conspire.py/tests/ /usr/conspire/python/
RUN rm -r conspire.py/
