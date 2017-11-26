package nn

import "math"

//Sigmoid: Sigmoid Activation Function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//DSigmoid: Sigmoid Activation Function Derivative
func DSigmoid(x float64) float64 {
	return x * (1 - x)
}

//Relu: ReLU Activation Function
func Relu(x float64) float64 {
	return math.Max(0, x)
}

//DRelu: ReLU Activation Function Derivative
func DRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

//TanH: TanH Activation Function
func TanH(x float64) float64 {
	return math.Tanh(x)
}

//DTanH: TanH Activation Function Derivative
func DTanH(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}
