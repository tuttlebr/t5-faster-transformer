#!/bin/bash
clear
perf_analyzer -m fastertransformer \
    --async \
    --percentile=95 \
    --concurrency-range 4:16:4 \
    --input-data /ft_workspace/perf_analyzer_data.json \
    -u 172.25.4.4:8001 \
    -i grpc \
    --measurement-interval 30000 \
    -v