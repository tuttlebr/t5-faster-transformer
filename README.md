# Deploying T5 with NVIDIA Triton Inference Server

My working repo for following the [NVIDIA blog post](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/) originally written by [Denis Timonin](https://developer.nvidia.com/blog/author/dtimonin/), [Bo Yang Hsueh](https://developer.nvidia.com/blog/author/bhsueh/), [Dhruv Singal](https://developer.nvidia.com/blog/author/dsingal/) and [Vinh Nguyen](https://developer.nvidia.com/blog/author/vinhn/)

![t5](https://developer-blogs.nvidia.com/wp-content/uploads/2022/07/image8.png)

## Get Running Fast(erTransformer)

You can modify the files below to tune any settings, including precision and GPU tensor parallelism:

```sh
workspace/00_clone_fastertransformer_backend.sh
workspace/01_clone_FasterTransformer_library.sh
workspace/02_download_t5_weights.sh
workspace/03_kernel_autotuning_t5.sh
workspace/04_start_triton_inference_server.sh
workspace/05_go_installer.sh && go run t5_go_client_demo.go
```

1. Build the triton fastertransformer backend: `docker compose build`
2. Compile the t5 model: `docker compose run faster-transformer`
3. Modify the subsequent `config.pbtxt` file, executable `workspace/03_kernel_autotuning_t5.sh` will print out what you need to chhange.
4. `docker compose up faster-transformer-server`
5. `docker compose up faster-transformer-client`

## Preliminary Results

Using NVIDIA's Perf Analyzer, run the most exhausting task, translation. You can run this from the interactive Jupyter environment by executing:

```sh
./perf_analyzer_runner.sh
```

```sh
text = "Translate English to German: He swung back the fishing pole and cast the line."
prediction = "Er schwenkte den Angelstab zurück und stieß die Angel."
```

```sh
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 30000 msec
  Using asynchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 4
  Pass [1] throughput: 8.69429 infer/sec. p95 latency: 458763 usec
  Pass [2] throughput: 8.7221 infer/sec. p95 latency: 458674 usec
  Pass [3] throughput: 8.7221 infer/sec. p95 latency: 458637 usec
  Client:
    Request count: 941
    Throughput: 8.71283 infer/sec
    p50 latency: 458238 usec
    p90 latency: 458520 usec
    p95 latency: 458693 usec
    p99 latency: 476971 usec
    Avg gRPC time: 458069 usec (marshal 3 usec + response wait 458065 usec + unmarshal 1 usec)
  Server:
    Inference count: 954
    Execution count: 954
    Successful request count: 954
    Avg request latency: 454737 usec (overhead 46 usec + queue 338436 usec + compute input 17 usec + compute infer 116190 usec + compute output 47 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 4, throughput: 8.71283 infer/sec, latency 458693 usec
```
