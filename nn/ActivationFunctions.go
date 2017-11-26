package nn

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func DSigmoid(x float64) float64 {
	return x * (1 - x)
}

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func DRelu(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func TanH(x float64) float64 {
	return math.Tanh(x)
}

func DTanH(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}
