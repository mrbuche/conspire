FROM python:latest
RUN curl https://sh.rustup.rs -fsSL | sh -s -- -y
RUN curl https://install.julialang.org -fsSL | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.juliaup/bin:${PATH}"
ARG VERSION
RUN if [ -z "$VERSION" ]; then \
        pip install git+https://github.com/mrbuche/conspire.py.git pytest; \
        julia -e 'using Pkg; Pkg.add(url="https://github.com/mrbuche/Conspire.jl")'; \
    else \
        pip install conspire==$VERSION pytest; \
        julia -e 'using Pkg; Pkg.add(name="Conspire", version="'$VERSION'")'; \
    fi
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
