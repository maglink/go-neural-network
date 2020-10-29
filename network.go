package gnn

import (
	"math"
	"math/rand"
	"sync/atomic"
)

type NeuralNetwork struct {
	Layers  []*Layer `json:"layers"`
	Options Options  `json:"options"`
}

type Layer struct {
	Type    LayerType      `json:"-"`
	Nodes   []*Node        `json:"nodes"`
	Network *NeuralNetwork `json:"-"`
}

type LayerType int

const (
	InputNetworkLayer LayerType = iota
	HiddenNetworkLayer
	OutputNetworkLayer
)

type Node struct {
	Id        uint64         `json:"id"`
	Links     []*Link        `json:"links"`
	BackLinks []*Link        `json:"back_links"`
	Network   *NeuralNetwork `json:"-"`
	Bias      float64        `json:"bias"`
}

type Link struct {
	FromId uint64  `json:"from"`
	ToId   uint64  `json:"to"`
	From   *Node   `json:"-"`
	To     *Node   `json:"-"`
	Weight float64 `json:"weight"`
}

func NewNeuralNetwork(opts Options) *NeuralNetwork {
	opts = setDefaultOptions(opts)
	nn := &NeuralNetwork{Options: opts}
	var nodeCounter uint64

	inputLayer := &Layer{Type: InputNetworkLayer}
	for i := 0; i < opts.InputsCount; i++ {
		inputLayer.Nodes = append(inputLayer.Nodes, &Node{
			Id:      atomic.AddUint64(&nodeCounter, 1),
			Network: nn,
			Bias:    randomBias(),
		})
	}
	nn.Layers = append(nn.Layers, inputLayer)

	for i := 0; i < len(opts.HiddenLayersSizes); i++ {
		hidLayer := &Layer{Type: HiddenNetworkLayer}
		for j := 0; j < opts.HiddenLayersSizes[i]; j++ {
			midLayerNode := &Node{
				Id:      atomic.AddUint64(&nodeCounter, 1),
				Network: nn,
				Bias:    randomBias(),
			}
			if i == 0 {
				for _, inputNode := range inputLayer.Nodes {
					link := &Link{
						FromId: inputNode.Id,
						ToId:   midLayerNode.Id,
						From:   inputNode,
						To:     midLayerNode,
						Weight: randomWeight(),
					}

					inputNode.Links = append(inputNode.Links, link)
					midLayerNode.BackLinks = append(midLayerNode.BackLinks, link)
				}
			} else {
				for _, prevMidLayerNode := range nn.Layers[i].Nodes {
					link := &Link{
						FromId: prevMidLayerNode.Id,
						ToId:   midLayerNode.Id,
						From:   prevMidLayerNode,
						To:     midLayerNode,
						Weight: randomWeight(),
					}

					prevMidLayerNode.Links = append(prevMidLayerNode.Links, link)
					midLayerNode.BackLinks = append(midLayerNode.BackLinks, link)
				}
			}
			hidLayer.Nodes = append(hidLayer.Nodes, midLayerNode)
		}
		nn.Layers = append(nn.Layers, hidLayer)
	}

	outputLayer := &Layer{Type: OutputNetworkLayer}
	for i := 0; i < opts.OutputsCount; i++ {
		outputNode := &Node{
			Id:      atomic.AddUint64(&nodeCounter, 1),
			Network: nn,
			Bias:    randomBias(),
		}
		for _, leftNode := range nn.Layers[len(nn.Layers)-1].Nodes {
			link := &Link{
				FromId: leftNode.Id,
				ToId:   outputNode.Id,
				From:   leftNode,
				To:     outputNode,
				Weight: randomWeight(),
			}
			leftNode.Links = append(leftNode.Links, link)
			outputNode.BackLinks = append(outputNode.BackLinks, link)
		}
		outputLayer.Nodes = append(outputLayer.Nodes, outputNode)
	}
	nn.Layers = append(nn.Layers, outputLayer)

	return nn
}

func (nn *NeuralNetwork) useWithFullResult(inputValues []float64) ([]float64, map[uint64]float64) {
	if len(nn.Layers) == 0 {
		return nil, nil
	}
	if len(inputValues) != len(nn.Layers[0].Nodes) {
		return nil, nil
	}

	valuesMap := map[uint64]float64{}
	for i, node := range nn.Layers[0].Nodes {
		valuesMap[node.Id] = inputValues[i]
	}

	result := make([]float64, nn.Options.OutputsCount)
	for layerIndex, layer := range nn.Layers {
		if layerIndex == 0 {
			continue
		}
		for _, node := range layer.Nodes {
			for _, link := range node.BackLinks {
				valuesMap[node.Id] += valuesMap[link.FromId]*link.Weight + node.Bias
			}
			valuesMap[node.Id] = nn.Options.ActivationFunc(valuesMap[node.Id])
		}
		if layerIndex == len(nn.Layers)-1 {
			for i, node := range layer.Nodes {
				result[i] = valuesMap[node.Id]
			}
		}
	}

	return result, valuesMap
}

func (nn *NeuralNetwork) Use(inputValues []float64) (outputValues []float64) {
	outputValues, _ = nn.useWithFullResult(inputValues)
	return
}

func (nn *NeuralNetwork) BackPropagation(inputValues, targetValues []float64) {
	outputValues, valuesMap := nn.useWithFullResult(inputValues)

	errorValues := map[uint64]float64{}
	for i, node := range nn.Layers[len(nn.Layers)-1].Nodes {
		errorValues[node.Id] = targetValues[i] - outputValues[i]
	}
	for layerIndex := len(nn.Layers) - 2; layerIndex > 0; layerIndex-- {
		for _, node := range nn.Layers[layerIndex].Nodes {
			for _, link := range node.Links {
				errorValues[node.Id] += errorValues[link.ToId] * link.Weight
			}
		}
	}

	for layerIndex, layer := range nn.Layers {
		if layerIndex == 0 {
			continue
		}

		for _, node := range layer.Nodes {
			for _, link := range node.BackLinks {
				link.Weight += nn.Options.LearningRate * errorValues[node.Id] * nn.Options.ActivationFuncDerivative(valuesMap[node.Id]) * valuesMap[link.FromId]
			}
		}
	}
}

type Options struct {
	InputsCount              int                         `json:"inputs_count"`
	OutputsCount             int                         `json:"outputs_count"`
	HiddenLayersSizes        []int                       `json:"hidden_layers_sizes"`
	ActivationFunc           func(value float64) float64 `json:"-"`
	ActivationFuncDerivative func(value float64) float64 `json:"-"`
	LearningRate             float64                     `json:"learning_rate"`
}

func Sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -value))
}
func SigmoidDerivative(value float64) float64 {
	return value * (1 - value)
}

func setDefaultOptions(opts Options) Options {
	if opts.InputsCount == 0 {
		opts.InputsCount = 5
	}
	if len(opts.HiddenLayersSizes) == 0 {
		opts.HiddenLayersSizes = append(opts.HiddenLayersSizes, 15, 7)
	}
	if opts.OutputsCount == 0 {
		opts.OutputsCount = 2
	}
	if opts.ActivationFunc == nil {
		opts.ActivationFunc = Sigmoid
	}
	if opts.ActivationFuncDerivative == nil {
		opts.ActivationFuncDerivative = SigmoidDerivative
	}
	if opts.LearningRate <= 0 || opts.LearningRate > 1 {
		opts.LearningRate = 0.5
	}
	return opts
}

func randomWeight() float64 {
	return 0.5 - rand.Float64()
}

func randomBias() float64 {
	return 0
}
