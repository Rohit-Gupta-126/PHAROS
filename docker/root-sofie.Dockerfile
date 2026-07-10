# ROOT built with TMVA SOFIE (ONNX parser) enabled.
#
# WHY THIS EXISTS: rootproject/root:latest (ROOT 6.38.00) ships the SOFIE
# runtime (libROOTTMVASofie.so) but NOT the ONNX parser
# (libROOTTMVASofieParser.so) -- the parser needs protobuf at build time and
# the official image is configured without it (see
# reports/phase2/sofie_probe.txt).
#
# WARNING -- DO NOT BUILD ON THE 16 GB DEV LAPTOP. A ROOT source build takes
# multiple hours and >8 GB RAM even with a trimmed component set. Run this
# once on a lab machine / CI runner with ample cores, then either push the
# image or just copy out the generated encoder header:
#
#   docker build -f docker/root-sofie.Dockerfile -t pharos/root-sofie .
#   docker run --rm -v "$PWD:/work" -w /work pharos/root-sofie \
#       python3 scripts/sofie_probe.py
#   # -> services/inference_sofie/generated/encoder_mu.hxx + .dat
#
# Until then, the runnable PHAROS trigger-inference path is ONNX Runtime
# (services/scorers/physics_scorer_sofie --backend ort).

FROM ubuntu:24.04 AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git python3-dev python3-numpy \
    libssl-dev libx11-dev libxext-dev libxft-dev libxpm-dev \
    libprotobuf-dev protobuf-compiler libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

ARG ROOT_VERSION=v6-38-00
RUN git clone --branch ${ROOT_VERSION} --depth 1 \
        https://github.com/root-project/root.git /src/root

# Minimal build: TMVA + SOFIE (+ its ONNX parser via builtin protobuf check),
# everything GUI/IO-exotic off to keep the build as small as possible.
RUN cmake -S /src/root -B /build \
        -DCMAKE_INSTALL_PREFIX=/opt/root \
        -DCMAKE_BUILD_TYPE=Release \
        -Dtmva=ON -Dtmva-sofie=ON -Dtmva-cpu=ON \
        -Dtmva-gpu=OFF -Dtmva-pymva=OFF \
        -Dgui=OFF -Dwebgui=OFF -Dx11=OFF -Dopengl=OFF \
        -Droofit=OFF -Dminuit2=ON -Ddataframe=ON \
    && cmake --build /build --target install -j"$(nproc)"

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-numpy libblas3 liblapack3 libprotobuf32t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/root /opt/root
ENV ROOTSYS=/opt/root \
    PATH=/opt/root/bin:$PATH \
    LD_LIBRARY_PATH=/opt/root/lib \
    PYTHONPATH=/opt/root/lib
CMD ["root", "-b"]
