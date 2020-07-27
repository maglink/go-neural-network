package main

import (
	"github.com/maglink/go-neural-network"
)

func main() {
	nn := gnn.NewNeuralNetwork(gnn.Options{InputsCount: 12, MiddleLayerSizes: []int{36, 64, 16}})
	println("Listen on http://localhost:3000/")
	nn.ListenAddr(":3000")
}
