package main

import (
	"fmt"
	"./nn"
	"./gomnist"
	"time"
	"encoding/csv"
	"os"
	"strconv"
)

func main() {

	trainingImgs, testImgs, trainLabels, testLabels := loadMNISTData()

	fmt.Print("Creating Example Sets...")

	trainingExampleSet := createExampleSet(trainingImgs, trainLabels)
	testingExampleSet := createExampleSet(testImgs, testLabels)

	fmt.Println("Done")

	fmt.Println("Creating output CSV File...Done")

	file, err := os.Create("data.csv")
	if err != nil {
		fmt.Println(err)
	}

	writer := csv.NewWriter(file)

	fmt.Println("Running Experiments...")

	records := [][]string{}

	for i := 59500; i < 60000; i += 100 {
		for j := 1; j < 11; j++ {
			fmt.Println(">[ Training Examples", i, "Hidden Layers", j, "]<")
			time, accuracy, trainingError := runTestIteration(0.2, 0.4, j, 10, trainingExampleSet, testingExampleSet, i)

			tmp := []string{}

			tmp = append(tmp, IntToString(i))
			tmp = append(tmp, IntToString(j))
			tmp = append(tmp, FloatToString(accuracy))
			tmp = append(tmp, Int64ToString(time))

			for _, n := range trainingError {
				tmp = append(tmp, FloatToString(n))
			}

			records = append(records, tmp)
		}
		writer.WriteAll(records)
		records = [][]string{}
	}

	writer.WriteAll(records)

	fmt.Println("Done Experiments")

	file.Close()
	writer.Flush()

}

func FloatToString(in float64) string {
	return strconv.FormatFloat(in, 'f', 6, 64)
}

func IntToString(in int) string {
	return strconv.Itoa(in)
}

func Int64ToString(in int64) string {
	return strconv.FormatInt(in, 10)
}

func runTestIteration(lr, mFactor float64, hiddens, epoch int, trainingData, testingData []nn.Example, numExamples int) (int64, float64, []float64) {

	network := nn.InitializeNewNetwork(lr, 784, hiddens, 10, epoch, mFactor)

	startingTime := time.Now().UnixNano()

	trainingError := network.TrainNetwork(trainingData[:numExamples])

	time := time.Now().UnixNano() - startingTime

	accuracy := (1 - network.EvaluateNetwork(testingData)) * 100

	return time, accuracy, trainingError

}

func normalizeImages(in []gomnist.RawImage) [][]float64 {
	arr := [][]float64{}
	for i := 0; i < len(in); i++ {
		tmp := []float64{}
		for j := 0; j < len(in[i]); j++ { tmp = append(tmp, float64(in[i][j]) / 255) }
		arr = append(arr, tmp)
	}
	return arr
}

func normalizeLabels(in []gomnist.Label) []int {
	arr := []int{}

	for i := 0; i < len(in); i++ {
		arr = append(arr, int(in[i]))
	}
	return arr
}

func GenerateDesiredOutput(n int, l int) []float64 {
	out := []float64{}
	for i := 0; i < l; i++ {
		if i == n {
			out = append(out, 1)
		} else {
			out = append(out, 0)
		}
	}
	return out
}

func createExampleSet(in []gomnist.RawImage, out []gomnist.Label) []nn.Example {

	input := normalizeImages(in)
	output := normalizeLabels(out)


	if len(in) != len(out) {
		fmt.Println("Error While Making Example Set")
	}

	examples := []nn.Example{}

	for i := 0; i < len(in); i++ {

		example := nn.Example{
			Input:  input[i],
			Output: GenerateDesiredOutput(output[i], 10),
		}

		examples = append(examples, example)
	}

	return examples
}

func loadMNISTData() ([]gomnist.RawImage, []gomnist.RawImage, []gomnist.Label, []gomnist.Label){

	fmt.Print("Loading Training Images & Labels...")

	_, _, trainingImgs, trainImgErr := gomnist.ReadImageFile("data/train-images-idx3-ubyte.gz")
	trainLabels, trainLabelErr := gomnist.ReadLabelFile("data/train-labels-idx1-ubyte.gz")

	fmt.Println("Done")

	fmt.Print("Loading Test Images & Labels...")

	_, _, testImgs, testImgErr := gomnist.ReadImageFile("data/t10k-images-idx3-ubyte.gz")
	testLabels, testLabelErr := gomnist.ReadLabelFile("data/t10k-labels-idx1-ubyte.gz")

	fmt.Println("Done")

	if trainImgErr != nil || trainLabelErr != nil || testImgErr != nil || testLabelErr != nil {
		fmt.Println("Error Loading MNIST Data.")
	}

	return trainingImgs, testImgs, trainLabels, testLabels


}
