package main

import (
	"fmt"
	"github.com/maglink/go-neural-network"
)

func main() {
	nn := gnn.NewNeuralNetwork(gnn.Options{})
	println(fmt.Sprintf("%#+v", nn.Use([]float64{1,1})))
	println(nn.ToString())
}
