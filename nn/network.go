package nn

import (
	"math/rand"
	"math"
	"time"
)

func GetRandomValue(a, b float64) float64 {
	return (b-a) * rand.Float64() + a
}

func CreateMatrix(i, j int) [][]float64 {
	m := make([][]float64, i)
	for a := 0; a < i; a++ {
		m[a] = make([]float64, j)
	}
	return m
}

func CreateVector(i int, fill float64) []float64 {
	v := make([]float64, i)
	for a := 0; a < i; a++ {
		v[a] = fill
	}
	return v
}

func Sigmoid(n float64, d bool) float64 {
	if d { return n * (1 - n) }
	return 1 / (1 + math.Exp(-n)) //possibly wrong derivative
}

func SquaredError(desired, actual []float64) float64 {
	if len(desired) != len(actual) {
		panic("Size of Desired and Actual Arrays are Not Equal.")
	}

	var sum float64 = 0
	for i := range desired { sum += math.Pow(desired[i] - actual[i], 2) }
	return 0.5 * sum
}

type Network struct {
	NInputs, NHidden, NOutputs, Epochs                     int
	InputActivations, HiddenActivations, OutputActivations []float64
	InputWeights, OutputWeights                            [][]float64
	InputChanges, OutputChanges                            [][]float64
	LearningRate, MFactor                                  float64
}

type Example struct {
	Input, Output []float64
}

func InitializeNewNetwork(learningRate float64, inputs, hiddens, outputs, epochCount int, mFactor float64) Network {

	rand.Seed(time.Now().UnixNano())

	inputs++
	hiddens++

	network := Network{
		NInputs:           inputs,
		NHidden:           hiddens,
		NOutputs:          outputs,
		InputActivations:  CreateVector(inputs, 1.0),
		HiddenActivations: CreateVector(hiddens, 1.0),
		OutputActivations: CreateVector(outputs, 1.0),
		InputWeights:      CreateMatrix(inputs, hiddens),
		OutputWeights:     CreateMatrix(hiddens, outputs),
		InputChanges:      CreateMatrix(inputs, hiddens),
		OutputChanges:     CreateMatrix(hiddens, outputs),
		LearningRate:      learningRate,
		MFactor:           mFactor,
		Epochs:            epochCount,
	}

	for i := 0; i < network.NInputs; i++ {
		for j := 0; j < network.NHidden; j++ {
			network.InputWeights[i][j] = GetRandomValue(-1, 1)
		}
	}

	for i := 0; i < network.NHidden; i++ {
		for j := 0; j < network.NOutputs; j++ {
			network.OutputWeights[i][j] = GetRandomValue(-1, 1)
		}
	}

	return network

}

func (network *Network) ForwardPropagate(input []float64) []float64 {
	if len(input) != network.NInputs-1 {
		panic("Error. Invalid Size of Inputs while Forward Propagating.")
	}

	for i := 0; i < network.NInputs-1; i++ {
		network.InputActivations[i] = input[i]
	}

	for i := 0; i < network.NHidden-1; i++ {
		var sum float64

		for j := 0; j < network.NInputs; j++ {
			sum += network.InputActivations[j] * network.InputWeights[j][i]
		}

		network.HiddenActivations[i] = Sigmoid(sum, false)
	}

	for i := 0; i < network.NOutputs; i++ {
		var sum float64

		for j := 0; j < network.NHidden; j++ {
			sum += network.HiddenActivations[j] * network.OutputWeights[j][i]
		}

		network.OutputActivations[i] = Sigmoid(sum, false)
	}

	return network.OutputActivations
}

func (network *Network) BackPropagate(targets []float64) float64 {
	if len(targets) != network.NOutputs {
		panic("Error. Invalid Size of Outputs while Backward Propagating.")
	}

	//Calculate Deltas for Each Layer

	outputDeltas := CreateVector(network.NOutputs, 0.0)
	for i := 0; i < network.NOutputs; i++ {
		outputDeltas[i] = Sigmoid(network.OutputActivations[i], true) * (targets[i] - network.OutputActivations[i])
	}

	hiddenDeltas := CreateVector(network.NHidden, 0.0)
	for i := 0; i < network.NHidden; i++ {
		var e float64

		for j := 0; j < network.NOutputs; j++ {
			e += outputDeltas[j] * network.OutputWeights[i][j]
		}

		hiddenDeltas[i] = Sigmoid(network.HiddenActivations[i], true) * e
	}

	//Adjust the Weights Accordingly

	for i := 0; i < network.NHidden; i++ {
		for j := 0; j < network.NOutputs; j++ {
			change := outputDeltas[j] * network.HiddenActivations[i]
			network.OutputWeights[i][j] = network.OutputWeights[i][j] + network.LearningRate*change + network.MFactor*network.OutputChanges[i][j]
			network.OutputChanges[i][j] = change
		}
	}

	for i := 0; i < network.NInputs; i++ {
		for j := 0; j < network.NHidden; j++ {
			change := hiddenDeltas[j] * network.InputActivations[i]
			network.InputWeights[i][j] = network.InputWeights[i][j] + network.LearningRate*change + network.MFactor*network.InputChanges[i][j]
			network.InputChanges[i][j] = change
		}
	}

	//Calculate Error

	var e float64

	for i := 0; i < len(targets); i++ {
		e += 0.5 * math.Pow(targets[i] - network.OutputActivations[i], 2)
	}

	return e
}

func (network *Network) TrainNetwork(examples []Example) []float64 {

	accuracy := make([]float64, network.Epochs)

	for i := 0; i < network.Epochs; i++ {
		var e float64
		for _, example := range examples {
			network.ForwardPropagate(example.Input)
			e += network.BackPropagate(example.Output)
		}
		accuracy[i] = 1 - (e / float64(len(examples)))
	}
	return accuracy
}

func (network *Network) EvaluateNetwork(examples []Example) float64 {

	var err float64

	for _, example := range examples {
		err += SquaredError(example.Output, network.ForwardPropagate(example.Input))
	}

	return err / float64(len(examples))

}

