#!/bin/bash

git clone https://github.com/NVIDIA/FasterTransformer.git
cd FasterTransformer \
    && git checkout 6b3fd4392831f972d48127e881a048567dd92811 \
    && mkdir -p build \
    && cd build \
    && git submodule init && git submodule update \
    && cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON .. \
    && make -j$(nproc)

ls -l /workspace/fastertransformer_backend/all_models/t5
