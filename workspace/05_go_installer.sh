#!/bin/bash
clear
curl -OL https://golang.org/dl/go1.18.9.linux-amd64.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xvf go1.18.9.linux-amd64.tar.gz
rm go1.18.9.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
go version

rm -rf common
git clone https://github.com/triton-inference-server/common.git

PACKAGE="grpc-client"

# Clean up before
rm -rf ./core

# Get the proto files from the common repo
mkdir core && cp common/protobuf/*.proto core/.

for i in ./core/*.proto
do 
  # https://developers.google.com/protocol-buffers/docs/reference/go-generated#package
  echo "option go_package = \"./${PACKAGE}\";" >> $i
done

# Requires protoc and protoc-gen-go plugin: https://grpc.io/docs/protoc-installation/
apt update
apt install -y protobuf-compiler
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2
export PATH="$PATH:$(go env GOPATH)/bin"
protoc -I core --go-grpc_out="./" --go_out="./" core/*.proto
echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.profile

echo
echo "Installation complete! Please run 'source ~/.profile' to update PATH"
echo "to include the new location of go."