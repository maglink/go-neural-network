package gnn

import "fmt"

func (nn *NeuralNetwork) ToString() string {
	if nn == nil {
		return ""
	}

	var result string
	for i, mLayer := range nn.Layers {
		switch mLayer.Type {
		case InputNetworkLayer:
			result += "input   \t{"
		case MiddleNetworkLayer:
			result += fmt.Sprintf("middle %d\t{", i)
		case OutputNetworkLayer:
			result += "output  \t{"
		}
		for i, _ := range mLayer.Nodes {
			result += fmt.Sprintf("[%d]", i)
		}
		result += "}\n"
	}

	return result
}
