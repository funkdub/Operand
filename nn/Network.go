package nn

import (
	"math/rand"
	"time"
	"math"
	"fmt"
)

//ConnectedNetwork: Structure to hold network data
type ConnectedNetwork struct {
	Structure []int
	Weights [][][]float64
	Activations, Deltas, Changes [][]float64
	LearningRate, MomentumFactor float64
	EpochCount int
	ActivationFunction, ActivationFunctionDerivative func(float64) float64
}

//Example: Structure to format examples correctly for network training
type Example struct {
	Input []float64
	Output []float64
}

//CreateVector: Function to fill an array of size i with value fill
func CreateVector(i int, fill float64) []float64 {
	v := make([]float64, i)
	for i := 0; i < i; i++ {
		v[i] = fill
	}
	return v
}

//InitializeNewNetwork: Initialize a new fully connected network
func InitializeNewNetwork(structure []int, learningRate, momentumFactor float64, epochCount int, activationFunction, activationFunctionDerivative func(float64) float64) ConnectedNetwork{

	rand.Seed(time.Now().UnixNano())

	//Initialize the network object
	n := ConnectedNetwork{
		Structure:                    structure,
		Weights:                      [][][]float64{},
		Activations:                  [][]float64{},
		Deltas:						  [][]float64{},
		Changes:   				      [][]float64{},
		LearningRate:                 learningRate,
		MomentumFactor:               momentumFactor,
		EpochCount:                   epochCount,
		ActivationFunction:           activationFunction,
		ActivationFunctionDerivative: activationFunctionDerivative,
	}

	tmp := []float64{}
	for i := 0; i < structure[0]; i++ {
		tmp = append(tmp, 0)
	}
	n.Activations = append(n.Activations, tmp)
	n.Changes = append(n.Changes, tmp)

	//Initialize the weights of the network to random values
	for i := 1; i < len(n.Structure); i++ {
		temp := [][]float64{}
		tempa := []float64{}
		for j := 0; j < n.Structure[i]; j++ {
			tempSec := []float64{}
			tempa = append(tempa, float64(0))
			for k := 0; k < n.Structure[i-1]; k++ { tempSec = append(tempSec, rand.Float64()) }
			temp = append(temp, tempSec)
		}
		n.Weights = append(n.Weights, temp)
		n.Activations = append(n.Activations, tempa)
		n.Changes = append(n.Changes, CreateVector(len(tempa), 0.0))
		n.Deltas = append(n.Deltas, CreateVector(len(tempa), 0.0))
	}
	return n
}

//ForwardPropagateInput: Calculate activation values for each neuron and return output layer outputs
func (n *ConnectedNetwork) ForwardPropagateInput(in []float64) []float64 {

	if len(in) != n.Structure[0] { panic("Invalid length of input relative to network input layer size during forward propagation.") }

	arr := in
	temp := []float64{}

	for i, val := range in {
		n.Activations[0][i] = val
	}

	for i := range n.Weights {
		for j := range n.Weights[i] {
			sum := float64(0)
			for k := range n.Weights[i][j] { sum += arr[k] * n.Weights[i][j][k] }
			newVal := n.ActivationFunction(sum)
			n.Activations[i+1][j] = newVal
			temp = append(temp, newVal)
		}
		arr = temp
		temp = []float64{}
	}
	return arr

}

//CalculateDeltas: A function to calculate the deltas of each neuron in the network.
func (n *ConnectedNetwork) CalculateDeltas(example Example) {

	if n.Structure[0] != len(example.Input) { panic("Input vector does not match size of network input layer.") }

	if n.Structure[len(n.Structure) - 1] != len(example.Output) { panic("Output vector does not match size of network output layer.") }

	n.ForwardPropagateInput(example.Input)

	index := len(n.Deltas) - 1

	for i := 0; i < len(n.Deltas[index]); i++ {
		n.Deltas[index][i] = n.ActivationFunctionDerivative(n.Activations[len(n.Activations) - 1][i]) * (example.Output[i] - n.Activations[len(n.Activations) - 1][i])
	}

	for delta := len(n.Deltas) - 2; delta >= 0; delta-- {
		for i := 0; i < n.Structure[delta+1]; i++ {
			var err float64
			for j := 0; j < n.Structure[delta+2]; j++ {
				err += n.Deltas[delta+1][j] * n.Weights[delta+1][j][i]
			}
			n.Deltas[delta][i] = n.ActivationFunctionDerivative(n.Activations[delta+1][i]) * err
		}
	}
}

//BackPropagate: A function to update weights using back propagation algorithm
func (n *ConnectedNetwork) BackPropagate(example Example) float64 {

	n.CalculateDeltas(example)

	for i := 1; i < len(n.Structure) - 1; i++ {
		for j := 0; j < n.Structure[i]; j++ {
			for k := 0; k < n.Structure[i+1]; k++ {
				ch := n.Deltas[i][k] * n.Activations[i][j]
				n.Weights[i][k][j] += n.LearningRate * ch + n.MomentumFactor*n.Changes[i][j]
				n.Changes[i][j] = ch
			}
		}
	}

	var err float64

	for i := 0; i < len(example.Output); i++ {
		err += 0.5 * math.Pow(example.Output[i] - n.Activations[len(n.Activations)-1][i], 2)
	}

	return err

}

//TrainNetwork: A function to train the network on an example set
func (n *ConnectedNetwork) TrainNetwork(examples []Example, debug bool) []float64 {

	accuracy := make([]float64, n.EpochCount)

	for i := 0; i < n.EpochCount; i++ {
		var e float64
		for _, example := range examples {
			e += n.BackPropagate(example)
		}
		accuracy[i] = (1 - (e / float64(len(examples)))) * 100

		if debug {
			fmt.Println("Epoch", i, ". Accuracy Percentage Achieved:", accuracy[i])
		}
	}
	return accuracy
}


//EvaluateNetwork: A function to evaluate a network on an example set
func (n *ConnectedNetwork) EvaluateNetwork(examples []Example) float64 {

	var err float64

	for _, example := range examples {
		err += SquaredError(example.Output, n.ForwardPropagateInput(example.Input))
	}

	return (1 - err / float64(len(examples))) * 100

}





