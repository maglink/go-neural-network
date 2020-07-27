package gnn

import (
	"math/rand"
	"sync/atomic"
)

type NeuralNetwork struct {
	Layers  []*Layer `json:"layers"`
	Options Options `json:"options"`
	onUpdate func()
}

type Layer struct {
	Type    LayerType `json:"-"`
	Nodes   []*Node `json:"nodes"`
	Network *NeuralNetwork `json:"-"`
}

type LayerType int

const (
	InputNetworkLayer LayerType = iota
	MiddleNetworkLayer
	OutputNetworkLayer
)

type Node struct {
	Id uint64 `json:"id"`
	Links   []*Link `json:"links"`
	BackLinks   []*Link `json:"back_links"`
	Network *NeuralNetwork `json:"-"`
}

type Link struct {
	FromId uint64 `json:"from"`
	ToId uint64 `json:"to"`
	From   *Node `json:"-"`
	To     *Node `json:"-"`
	Weight float64 `json:"weight"`
}

func NewNeuralNetwork(opts Options) *NeuralNetwork {
	opts = setDefaultOptions(opts)
	nn := &NeuralNetwork{Options: opts}
	var nodeCounter uint64

	inputLayer := &Layer{Type: InputNetworkLayer}
	for i := 0; i < opts.InputsCount; i++ {
		inputLayer.Nodes = append(inputLayer.Nodes, &Node{Id: atomic.AddUint64(&nodeCounter, 1)})
	}
	nn.Layers = append(nn.Layers, inputLayer)

	for i := 0; i < len(opts.MiddleLayerSizes); i++ {
		midLayer := &Layer{Type: MiddleNetworkLayer}
		for j := 0; j < opts.MiddleLayerSizes[i]; j++ {
			midLayerNode := &Node{Id: atomic.AddUint64(&nodeCounter, 1)}
			if i == 0 {
				for _, inputNode := range inputLayer.Nodes {
					inputNode.Links = append(inputNode.Links, &Link{
						FromId: inputNode.Id,
						ToId: midLayerNode.Id,
						From: inputNode,
						To: midLayerNode,
						Weight: rand.Float64(),
					})
					midLayerNode.BackLinks = append(midLayerNode.Links, &Link{
						FromId: inputNode.Id,
						ToId: midLayerNode.Id,
						From: inputNode,
						To: midLayerNode,
						Weight: rand.Float64(),
					})
				}
			} else {
				for _, prevMidLayerNode := range nn.Layers[i].Nodes {
					prevMidLayerNode.Links = append(prevMidLayerNode.Links, &Link{
						FromId: prevMidLayerNode.Id,
						ToId: midLayerNode.Id,
						From: prevMidLayerNode,
						To: midLayerNode,
						Weight: rand.Float64(),
					})
					midLayerNode.BackLinks = append(midLayerNode.Links, &Link{
						FromId: prevMidLayerNode.Id,
						ToId: midLayerNode.Id,
						From: prevMidLayerNode,
						To: midLayerNode,
						Weight: rand.Float64(),
					})
				}
			}
			midLayer.Nodes = append(midLayer.Nodes, midLayerNode)
		}
		nn.Layers = append(nn.Layers, midLayer)
	}

	outputLayer := &Layer{Type: OutputNetworkLayer}
	for i := 0; i < opts.OutputsCount; i++ {
		outputNode := &Node{Id: atomic.AddUint64(&nodeCounter, 1)}
		for _, leftNode := range nn.Layers[len(nn.Layers)-1].Nodes {
			leftNode.Links = append(leftNode.Links, &Link{
				FromId: leftNode.Id,
				ToId: outputNode.Id,
				From: leftNode,
				To: outputNode,
				Weight: rand.Float64(),
			})
			outputNode.BackLinks = append(outputNode.Links, &Link{
				FromId: leftNode.Id,
				ToId: outputNode.Id,
				From: leftNode,
				To: outputNode,
				Weight: rand.Float64(),
			})
		}
		outputLayer.Nodes = append(outputLayer.Nodes, outputNode)
	}
	nn.Layers = append(nn.Layers, outputLayer)

	return nn
}

func (nn *NeuralNetwork) Use(inputValues []float64) (outputValues []float64) {
	return nil
}

type Options struct {
	InputsCount      int `json:"inputs_count"`
	OutputsCount     int `json:"outputs_count"`
	MiddleLayerSizes []int `json:"middle_layer_sizes"`
	ActivateFunc     func(value float64) bool `json:"-"`
}

func DefaultActivateFunc(value float64) bool {
	return value >= 0.5
}

func setDefaultOptions(opts Options) Options {
	if opts.InputsCount == 0 {
		opts.InputsCount = 5
	}
	if len(opts.MiddleLayerSizes) == 0 {
		opts.MiddleLayerSizes = append(opts.MiddleLayerSizes, 15, 7)
	}
	if opts.OutputsCount == 0 {
		opts.OutputsCount = 2
	}
	if opts.ActivateFunc == nil {
		opts.ActivateFunc = DefaultActivateFunc
	}
	return opts
}
