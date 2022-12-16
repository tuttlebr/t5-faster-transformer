package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	triton "github.com/triton-inference-server/client/src/grpc_generated/go/grpc-client"
	"log"
	"time"

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

func ModelInferRequest(client triton.GRPCInferenceServiceClient, rawInput [][]byte, modelName string, modelVersion string) *triton.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "input_ids",
			Datatype: "UINT32",
			Shape:    []int64{1, 21},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "sequence_length",
			Datatype: "UINT32",
			Shape:    []int64{1, 1},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "max_output_len",
			Datatype: "UINT32",
			Shape:    []int64{1, 1},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "runtime_top_k",
			Datatype: "UINT32",
			Shape:    []int64{1, 1},
		},
	}

	// Create request input output tensors
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "output_ids",
		},
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "sequence_length",
		},
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "cum_log_probs",
		},
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "output_log_probs",
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := triton.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput[0])
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput[1])
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput[2])
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput[3])

	// Submit inference request to server
	modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

// Convert int32 input data into raw bytes (assumes Little Endian)
func Preprocess(inputs [][]int32) [][]byte {
	inputData0 := inputs[0]
	inputData1 := inputs[1]
	inputData2 := inputs[2]
	inputData3 := inputs[3]

	var inputBytes0 []byte
	var inputBytes1 []byte
	var inputBytes2 []byte
	var inputBytes3 []byte
	// Temp variable to hold our converted int32 -> []byte
	bs := make([]byte, 4)
	for i := 0; i < len(inputData0); i++ {
		binary.LittleEndian.PutUint32(bs, uint32(inputData0[i]))
		inputBytes0 = append(inputBytes0, bs...)
	}

	for i := 0; i < len(inputData1); i++ {
		binary.LittleEndian.PutUint32(bs, uint32(inputData1[i]))
		inputBytes1 = append(inputBytes1, bs...)
	}

	for i := 0; i < len(inputData2); i++ {
		binary.LittleEndian.PutUint32(bs, uint32(inputData2[i]))
		inputBytes2 = append(inputBytes2, bs...)
	}

	for i := 0; i < len(inputData3); i++ {
		binary.LittleEndian.PutUint32(bs, uint32(inputData3[i]))
		inputBytes3 = append(inputBytes3, bs...)
	}

	return [][]byte{inputBytes0, inputBytes1, inputBytes2, inputBytes3}
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

	inputData0 := []int32{30355, 15, 1566, 12, 2968, 10, 216, 3, 7, 210, 425, 223, 8, 5095, 11148, 11, 4061, 8, 689, 5, 1}
	inputData1 := []int32{21}
	inputData2 := []int32{128}
	inputData3 := []int32{1}

	inputs := [][]int32{inputData0, inputData1, inputData2, inputData3}
	rawInput := Preprocess(inputs)

	inferResponse := ModelInferRequest(client, rawInput, FLAGS.ModelName, FLAGS.ModelVersion)

	outputs := Postprocess(inferResponse)
	outputData0 := outputs[0]
	outputData1 := outputs[1]

	fmt.Println("\nChecking Inference Outputs\n--------------------------")
	fmt.Println("\noutput_ids:")
	fmt.Println(outputData0[0:outputData1[0]])
	fmt.Println("\nsequence_length:")
	fmt.Println(outputData1)

}
