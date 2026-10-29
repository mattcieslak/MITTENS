FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN pip install --no-cache-dir "maturin>=1.7,<2"
COPY . /src/mittens
WORKDIR /src/mittens
RUN maturin build --release --out /wheels

FROM python:3.12-slim
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
ENTRYPOINT ["python"]
