package operand

import (
	"math/rand"
	"math"
	"time"
	"fmt"
)

type NeuralNet struct {
	Structure []int
	Weights [][][]float64
	Biases [][]float64
	LearningRate float64
	EpochCount int
}

type Example struct {
	Input []float64
	Output []float64
}

func Relu(n float64, d bool) float64 {
	if d{
		if n > 0 {
			return 1
		} else { return 0 }
	}
	return math.Max(0, n)
}

func Softmax(arr []float64) []float64 {

	outArr := make([]float64, len(arr))

	for i, n := range arr {
		var sum float64 = 0
		for _, v := range arr { sum += math.Exp(v) }
		outArr[i] = math.Exp(n) / sum
	}

	return outArr
}

func Sigmoid(n float64, d bool) float64 {
	if d {
		return n * (1 - n)
	}
	return 1 / (1+math.Exp(-n))
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

func SquaredError(desired, actual []float64) float64 {
	if len(desired) != len(actual) {
		panic("Size of Desired and Actual Arrays are Not Equal.")
	}

	var sum float64 = 0
	for i := range desired { sum += math.Pow(desired[i] - actual[i], 2) }
	return 0.5 * sum
}

func (network *NeuralNet)CalculateSquaredErrorForExample(example Example) float64 {
	return SquaredError(example.Output, Softmax(network.ForwardPropagateInput(example.Input)))
}

func InitializeRandomNetwork(Structure []int, LearningRate float64, EpochCount int) NeuralNet{

	rand.Seed(time.Now().UnixNano())

	network := NeuralNet{
		Structure:    Structure,
		Weights:      nil,
		Biases:       nil,
		LearningRate: LearningRate,
		EpochCount:   EpochCount,
	}

	if len(network.Structure) == 0 {
		panic("The network must contain atleast one input, output and hidden layer.")
	}

	fmt.Print("Initalizing new Neural Network")

	network.Weights = [][][]float64{}
	network.Biases = [][]float64{}
	for i := 1; i < len(network.Structure); i++ {
		temp := [][]float64{}
		tempb := []float64{}
		for j := 0; j < network.Structure[i]; j++ {
			tempSec := []float64{}
			tempb = append(tempb, rand.Float64())
			for k := 0; k < network.Structure[i-1]; k++ { tempSec = append(tempSec, rand.Float64()) }
			temp = append(temp, tempSec)
		}
		network.Weights = append(network.Weights, temp)
		network.Biases = append(network.Biases, tempb)
		fmt.Print(".")
	}

	fmt.Println("Done")

	return network
}

func (network *NeuralNet) ForwardPropagateInput(in []float64) []float64 {

	arr := in
	temp := []float64{}

	if len(arr) != len(network.Weights[0][0]) {
		panic("Tried to Forward Propagate Value with Input Size Not Equal to Input Layer.")
	}

	for i := range network.Weights {
		for j := range network.Weights[i] {
			sum := network.Biases[i][j]
			for k := range network.Weights[i][j] { sum += arr[k] * network.Weights[i][j][k] }
			temp = append(temp, Sigmoid(sum, false))
		}
		arr = temp
		temp = []float64{}
	}
	return arr
}

func (network *NeuralNet) BackPropegateExample(example Example) []float64{

	if network.Structure[0] != len(example.Input) {
		panic("Input vector does not match size of network input layer.")
	}

	if network.Structure[len(network.Structure) - 1] != len(example.Output) {
		panic("Output vector does not match size of network output layer.")
	}

	aValues := [][]float64{}
	errs := [][]float64{}

	//Populate Activation Value and Error Delta Array with Arbitrary Values
	for i := 0; i < len(network.Structure); i++ {
		aValues = append(aValues, make([]float64, network.Structure[i]))
		if i != 0 { errs = append(errs, make([]float64, network.Structure[i])) }
	}

	//Set the last layer of activations to the true output of the network.
	aValues[len(aValues) - 1] = network.ForwardPropagateInput(example.Input)

	//Set the first layer of activations equal to the input
	aValues[0] = example.Input

	//Output Layer Error Delta Calculations
	for outputNeuron := 0; outputNeuron < network.Structure[len(network.Structure) - 1]; outputNeuron++ {
		errs[len(errs) - 1][outputNeuron] = example.Output[outputNeuron] - aValues[len(aValues) - 1][outputNeuron]
	}

	//Hidden Neuron Deltas

	previousLayerDeltas := errs[len(errs) - 1]

	for hLayer := len(network.Structure) - 2; hLayer > 0; hLayer-- {

		newDeltas := []float64{}

		for i := 0; i < network.Structure[hLayer]; i++ {

			var e float64

			for j := 0; j < len(previousLayerDeltas); j++ {

				fmt.Println("Testing", network.Weights[hLayer][i][j], "Index", j)

				e += previousLayerDeltas[j] * network.Weights[hLayer][i][j]
			}

			newDeltas = append(newDeltas, )


		}

	}



	fmt.Println("Desired Result", example.Output)
	fmt.Println("Actual Result", aValues[len(aValues) - 1])
	fmt.Println("Errors", errs)

	return errs[len(errs) - 1]

}


