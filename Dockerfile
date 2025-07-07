FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.juliaup/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        git clone https://github.com/mrbuche/conspire.py.git; \
        git clone https://github.com/mrbuche/Conspire.jl.git --recurse-submodules; \
        julia -e 'using Pkg; Pkg.develop(path="Conspire.jl"); Pkg.build("Conspire"); Pkg.precompile()'; \
    else \
        pip install conspire==$VERSION pytest; \
        git clone https://github.com/mrbuche/conspire.py.git --branch $VERSION; \
        git clone https://github.com/mrbuche/Conspire.jl.git --branch $VERSION --recurse-submodules; \
        julia -e 'using Pkg; Pkg.develop(path="Conspire.jl"); Pkg.build("Conspire"); Pkg.precompile()'; \
    fi
RUN mkdir -p /usr/conspire/python/
RUN mv conspire.py/pyproject.toml /usr/conspire/python/
RUN mv conspire.py/tests/ /usr/conspire/python/
RUN rm -r conspire.py/
