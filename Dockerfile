FROM continuumio/miniconda3:latest

ARG DEBIAN_FRONTEND=noninteractive
ARG PROCESSING_VERSION=3.5.4
ARG PROCESSING_URL=https://github.com/processing/processing/releases/download/processing-0270-${PROCESSING_VERSION}/processing-${PROCESSING_VERSION}-linux64.tgz

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROCESSING_HOME=/opt/processing \
    PROCESSING_JAVA=/opt/processing/processing-java \
    DISPLAY=:99

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        xvfb \
        xauth \
        libgl1 \
        libglu1-mesa \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxtst6 \
        libxi6 \
        libfreetype6 \
        libfontconfig1 \
        libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt \
    && curl -fsSL "${PROCESSING_URL}" -o /tmp/processing.tgz \
    && tar -xzf /tmp/processing.tgz -C /opt \
    && mv "/opt/processing-${PROCESSING_VERSION}" "${PROCESSING_HOME}" \
    && rm /tmp/processing.tgz \
    && test -x "${PROCESSING_JAVA}"

WORKDIR /workspace/RealPDEBench
COPY environment.yml pyproject.toml README.md ./
RUN conda env create --file environment.yml \
    && conda run --name realpdebench python -m pip install --no-cache-dir "gym==0.26.2" \
    && conda clean --all --yes

ENV PATH=/opt/conda/envs/realpdebench/bin:${PATH}

COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .

COPY docker-entrypoint.sh /usr/local/bin/realpdebench-entrypoint
RUN chmod +x /usr/local/bin/realpdebench-entrypoint

ENTRYPOINT ["realpdebench-entrypoint"]
CMD ["bash"]
