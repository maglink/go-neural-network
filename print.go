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
		case HiddenNetworkLayer:
			result += fmt.Sprintf("middle %d\t{", i)
		case OutputNetworkLayer:
			result += "output  \t{"
		}
		for i := range mLayer.Nodes {
			result += fmt.Sprintf("[%d]", i)
		}
		result += "}\n"
		for _, link := range mLayer.Nodes[len(mLayer.Nodes)/2].Links {
			result += fmt.Sprintf("(%f)", link.Weight)
		}

		result += "\n"
	}

	return result
}
