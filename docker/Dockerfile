FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Maximilian Drexler"
LABEL description="Privacy-aware construction site monitoring pipeline"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        screen \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN cd /workspace \
    && git clone https://github.com/IDEA-Research/GroundingDINO.git \
    && cd GroundingDINO \
    && pip install -e . \
    && mkdir -p weights \
    && wget -q -P weights \
       https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

VOLUME ["/workspace/data"]
VOLUME ["/workspace/privacy_pipeline"]
VOLUME ["/workspace/output"]

CMD ["/bin/bash"]
