# ci/sandbox.Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
    sympy==1.13.3 \
    scipy==1.14.1 \
    numpy==2.1.3 \
    z3-solver==4.13.3.0 \
    hypothesis==6.119.0 \
    pint==0.24.4

RUN (getent passwd 65534 >/dev/null || useradd --uid 65534 --no-create-home --shell /usr/sbin/nologin sandboxuser) && \
    mkdir -p /work
WORKDIR /work
USER 65534:65534
