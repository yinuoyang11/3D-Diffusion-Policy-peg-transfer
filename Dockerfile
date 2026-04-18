FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=appuser

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/3D-Diffusion-Policy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_train.txt /tmp/requirements_train.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements_train.txt

COPY 3D-Diffusion-Policy /workspace/3D-Diffusion-Policy/3D-Diffusion-Policy
RUN python -m pip install -e /workspace/3D-Diffusion-Policy/3D-Diffusion-Policy

RUN groupadd --gid ${GROUP_ID} ${USER_NAME} && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash ${USER_NAME} && \
    chown -R ${USER_NAME}:${USER_NAME} /workspace/3D-Diffusion-Policy

USER ${USER_NAME}
WORKDIR /workspace/3D-Diffusion-Policy/3D-Diffusion-Policy

CMD ["/bin/bash"]
