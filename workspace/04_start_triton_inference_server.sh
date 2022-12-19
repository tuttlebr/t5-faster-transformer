#!/bin/bash
LOG_FILENAME_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
TRITON_VERSION=$(cat /opt/tritonserver/TRITON_VERSION)
LOG_FILENAME="${LOG_FILENAME_TIMESTAMP}_tritonv${TRITON_VERSION}.log"

/opt/tritonserver/bin/tritonserver \
    --model-repository=triton-model-store/t5/ \
    --disable-auto-complete-config \
    --model-control-mode=poll \
    --repository-poll-secs=30 \
    --id="FasterTransformer-T5" \
    --log-verbose=3 \
    --log-file=${LOG_FILENAME}
    