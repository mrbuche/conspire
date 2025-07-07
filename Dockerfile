FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.juliaup/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        git clone --recurse-submodules https://github.com/mrbuche/Conspire.jl.git; \
        julia -e 'using Pkg; Pkg.develop(path="Conspire.jl"); Pkg.precompile()'; \
        git clone https://github.com/mrbuche/conspire.py.git; \
    else \
        pip install conspire==$VERSION pytest; \
        julia -e 'using Pkg; Pkg.add(name="Conspire", version="'$VERSION'"); Pkg.precompile()'; \
        git clone https://github.com/mrbuche/conspire.py.git --branch $VERSION; \
    fi
RUN mkdir -p /usr/conspire/python/
RUN mv conspire.py/pyproject.toml /usr/conspire/python/
RUN mv conspire.py/tests/ /usr/conspire/python/
RUN rm -r conspire.py/
