package gnn

type NeuralNetwork struct {
	Layers  []*Layer
	Options Options
}

type Layer struct {
	Type    LayerType
	Nodes   []*Node
	Network *NeuralNetwork
}

type LayerType int

const (
	InputNetworkLayer LayerType = iota
	MiddleNetworkLayer
	OutputNetworkLayer
)

type Node struct {
	Links   []*Link
	Network *NeuralNetwork
}

type Link struct {
	From   *Node
	To     *Node
	Weight float64
}

func NewNeuralNetwork(opts Options) *NeuralNetwork {
	opts = setDefaultOptions(opts)
	nn := &NeuralNetwork{Options: opts}

	inputLayer := &Layer{Type: InputNetworkLayer}
	for i := 0; i < opts.InputsCount; i++ {
		inputLayer.Nodes = append(inputLayer.Nodes, &Node{})
	}
	nn.Layers = append(nn.Layers, inputLayer)

	for i := 0; i < len(opts.MiddleLayerSizes); i++ {
		midLayer := &Layer{Type: MiddleNetworkLayer}
		for j := 0; j < opts.MiddleLayerSizes[i]; j++ {
			midLayerNode := &Node{}
			if i == 0 {
				for _, inputNode := range inputLayer.Nodes {
					inputNode.Links = append(inputNode.Links, &Link{
						From: inputNode,
						To: midLayerNode,
					})
				}
			} else {
				for _, prevMidLayerNode := range nn.Layers[i].Nodes {
					prevMidLayerNode.Links = append(prevMidLayerNode.Links, &Link{
						From: prevMidLayerNode,
						To: midLayerNode,
					})
				}
			}
			midLayer.Nodes = append(midLayer.Nodes, midLayerNode)
		}
		nn.Layers = append(nn.Layers, midLayer)
	}

	outputLayer := &Layer{Type: OutputNetworkLayer}
	for i := 0; i < opts.OutputsCount; i++ {
		outputNode := &Node{}
		for _, leftNode := range nn.Layers[len(nn.Layers)-1].Nodes {
			leftNode.Links = append(leftNode.Links, &Link{
				From: leftNode,
				To: outputNode,
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
	InputsCount      int
	OutputsCount     int
	MiddleLayerSizes []int
	ActivateFunc     func(value float64) bool
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
