package main

import (
	"fmt"
	"github.com/maglink/go-neural-network"
	"github.com/maglink/go-neural-network/utils"
	"github.com/maglink/idx-parser"
	"log"
)

func main() {
	nn := gnn.NewNeuralNetwork(gnn.Options{
		InputsCount: 784,
		HiddenLayersSizes: []int{16, 16},
		OutputsCount: 10,
		LearningRate: 0.05,
	})
	go training(nn)

	log.Println("Listen on http://localhost:3000/")
	nn.ListenAddr(":3000")
}

func training(nn *gnn.NeuralNetwork) {
	trainImages, trainLabels := prepareTrainData()
	checkImages, checkLabels := prepareCheckData()

	inputValues := make([]float64, 784)
	targetValues := make([]float64, 10)
	outputValues := make([]float64, 10)
	var j, label int

	for i := 0; i < trainImages.GetCount(); i++ {
		setInputValuesFromImage(inputValues, trainImages.GetImage(i))
		label = int(trainLabels.GetLabel(i))
		for j = 0; j < 10; j++ {
			if j == label {
				targetValues[j] = 1
			} else {
				targetValues[j] = 0
			}
		}

		nn.BackPropagation(inputValues, targetValues)
	}

	println("training is done")

	var all, success float64
	for i := 0; i < checkImages.GetCount(); i++ {
		setInputValuesFromImage(inputValues, checkImages.GetImage(i))
		outputValues = nn.Use(inputValues)
		if getMaxOutputIndex(outputValues) == int(checkLabels.GetLabel(i)) {
			success++
		}
		all++
	}

	println("success rate", fmt.Sprintf("%f", success/all))

	checkImages.SaveToFile(111, "img100.bmp")

	setInputValuesFromImage(inputValues, checkImages.GetImage(111))
	println(fmt.Sprintf("%#+v", nn.Use(inputValues)))
	println("label from lib", checkLabels.GetLabel(111))
}

func getMaxOutputIndex(outputValues []float64) (j int) {
	var m float64
	for i, e := range outputValues {
		if i==0 || e > m {
			m = e
			j = i
		}
	}
	return
}

func setInputValuesFromImage(inputValues []float64, imageData []byte) {
	for j := range imageData {
		inputValues[j] = float64(imageData[j]) / 255
	}
}

func prepareTrainData() (trainImages *idx_parser.IdxImages, trainLabels *idx_parser.IdxLabels) {
	trainingImagesFileName := "train-images.idx3-ubyte"
	trainingImagesFileNameGz := "train-images-idx3-ubyte.gz"
	err := utils.DownloadAndDecompress("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
		trainingImagesFileNameGz, trainingImagesFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	trainingLabelsFileName := "train-labels.idx1-ubyte"
	trainingLabelsFileNameGz := "train-labels-idx1-ubyte.gz"
	err = utils.DownloadAndDecompress("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
		trainingLabelsFileNameGz, trainingLabelsFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	trainImages, err = idx_parser.ReadImages(trainingImagesFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	trainLabels, err = idx_parser.ReadLabels(trainingLabelsFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	if trainImages.GetCount() != trainLabels.GetCount() {
		log.Fatal("counts of labels and images are not equal")
	}
	if trainImages.GetCount() == 0 {
		log.Fatal("count of labels or images is 0")
	}

	return
}

func prepareCheckData() (checkImages *idx_parser.IdxImages, checkLabels *idx_parser.IdxLabels) {
	checkingImagesFileName := "t10k-images.idx3-ubyte"
	checkingImagesFileNameGz := "t10k-images-idx3-ubyte.gz"
	err := utils.DownloadAndDecompress("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
		checkingImagesFileNameGz, checkingImagesFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	checkingLabelsFileName := "t10k-labels.idx1-ubyte"
	checkingLabelsFileNameGz := "t10k-labels-idx1-ubyte.gz"
	err = utils.DownloadAndDecompress("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
		checkingLabelsFileNameGz, checkingLabelsFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	checkImages, err = idx_parser.ReadImages(checkingImagesFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	checkLabels, err = idx_parser.ReadLabels(checkingLabelsFileName)
	if err != nil {
		log.Fatal(err.Error())
	}

	if checkImages.GetCount() != checkLabels.GetCount() {
		log.Fatal("counts of labels and images are not equal")
	}
	if checkImages.GetCount() == 0 {
		log.Fatal("count of labels or images is 0")
	}

	return
}