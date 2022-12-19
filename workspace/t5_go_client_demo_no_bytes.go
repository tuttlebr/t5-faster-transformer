package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"time"

	triton "github.com/triton-inference-server/client/src/grpc_generated/go/grpc-client"

	"google.golang.org/grpc"
)

type Flags struct {
	ModelName    string
	ModelVersion string
	BatchSize    int
	URL          string
}

func parseFlags() Flags {
	var flags Flags
	// https://github.com/NVIDIA/triton-inference-server/tree/master/docs/examples/model_repository/simple
	flag.StringVar(&flags.ModelName, "m", "fastertransformer", "Name of model being served. (Required)")
	flag.StringVar(&flags.ModelVersion, "1", "", "Version of model. Default: Latest Version.")
	flag.IntVar(&flags.BatchSize, "b", 0, "Batch size. Default: 0.")
	flag.StringVar(&flags.URL, "u", "172.25.4.4:8001", "Inference Server URL. Default: 172.25.4.4:8001")
	flag.Parse()
	return flags
}

func ServerLiveRequest(client triton.GRPCInferenceServiceClient) *triton.ServerLiveResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := triton.ServerLiveRequest{}
	// Submit ServerLive request to server
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		log.Fatalf("Couldn't get server live: %v", err)
	}
	return serverLiveResponse
}

func ServerReadyRequest(client triton.GRPCInferenceServiceClient) *triton.ServerReadyResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := triton.ServerReadyRequest{}
	// Submit ServerReady request to server
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		log.Fatalf("Couldn't get server ready: %v", err)
	}
	return serverReadyResponse
}

func ModelMetadataRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) *triton.ModelMetadataResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create status request for a given model
	modelMetadataRequest := triton.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	// Submit modelMetadata request to server
	modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		log.Fatalf("Couldn't get server model metadata: %v", err)
	}
	return modelMetadataResponse
}

func ModelInferRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) *triton.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		{
			Name:     "input_ids",
			Datatype: "UINT32",
			Shape:    []int64{1, 9},
			Contents: &triton.InferTensorContents{UintContents: []uint32{1, 3, 2, 20489, 3155, 1028, 35, 63, 1}}, //fixed contents for mimicking same request
		},
		{
			Name:     "sequence_length",
			Datatype: "UINT32",
			Shape:    []int64{1, 1},
			Contents: &triton.InferTensorContents{UintContents: []uint32{9}},
		},
		{
			Name:     "max_output_len",
			Datatype: "UINT32",
			Shape:    []int64{1, 1},
			Contents: &triton.InferTensorContents{UintContents: []uint32{128}},
		},
		{
			Name:     "runtime_top_k",
			Datatype: "UINT32",
			Shape:    []int64{1, 1},
			Contents: &triton.InferTensorContents{UintContents: []uint32{1}},
		},
	}

	// Create request input output tensors
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: "output_ids",
		},
		{
			Name: "sequence_length",
		},
		{
			Name: "cum_log_probs",
		},
		{
			Name: "output_log_probs",
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := &triton.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	// Submit inference request to server
	modelInferResponse, err := client.ModelInfer(ctx, modelInferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

// Convert slice of 4 bytes to int32 (assumes Little Endian)
func readInt32(fourBytes []byte) int32 {
	buf := bytes.NewBuffer(fourBytes)
	var retval int32
	binary.Read(buf, binary.LittleEndian, &retval)
	return retval
}

// Convert output's raw bytes into int32 data (assumes Little Endian)
func Postprocess(inferResponse *triton.ModelInferResponse) [][]int32 {
	outputBytes0 := inferResponse.RawOutputContents[0]
	outputBytes1 := inferResponse.RawOutputContents[1]

	outputData0 := make([]int32, len(outputBytes0)/4)
	outputData1 := make([]int32, len(outputBytes1)/4)

	for i := 0; i < len(outputData0); i++ {
		outputData0[i] = readInt32(outputBytes0[i*4 : i*4+4])
	}

	for i := 0; i < len(outputData1); i++ {
		outputData1[i] = readInt32(outputBytes1[i*4 : i*4+4])
	}

	return [][]int32{outputData0, outputData1}
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	// Connect to gRPC server
	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
	}
	defer conn.Close()

	// Create client from gRPC server connection
	client := triton.NewGRPCInferenceServiceClient(conn)

	serverLiveResponse := ServerLiveRequest(client)
	fmt.Printf("Triton Health - Live: %v\n", serverLiveResponse.Live)

	serverReadyResponse := ServerReadyRequest(client)
	fmt.Printf("Triton Health - Ready: %v\n", serverReadyResponse.Ready)

	fmt.Printf("Triton Metadata:\n")
	modelMetadataResponse := ModelMetadataRequest(client, FLAGS.ModelName, "")
	modelMetadataResponsePretty, _ := json.MarshalIndent(modelMetadataResponse, "", " ")
	fmt.Println(string(modelMetadataResponsePretty))

	inferResponse := ModelInferRequest(client, FLAGS.ModelName, FLAGS.ModelVersion)

	outputs := Postprocess(inferResponse)
	outputData0 := outputs[0]
	outputData1 := outputs[1]

	fmt.Println("\nChecking Inference Outputs\n--------------------------")
	fmt.Println("\noutput_ids:")
	fmt.Println(outputData0[0:outputData1[0]])
	fmt.Println("\nsequence_length:")
	fmt.Println(outputData1)

}
