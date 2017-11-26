package nn

import "math"

//GenerateExample A function used to generate an Operand.nn.Example instance.
func GenerateExample(in, out []float64) Example {
	return Example{
		Input:  in,
		Output: out,
	}
}

//SquaredError A function to calculate the error of a training/testing example.
func SquaredError(desired, actual []float64) float64 {
	if len(desired) != len(actual) {
		panic("Size of Desired and Actual Arrays are Not Equal.")
	}

	var sum float64
	for i := range desired {
		sum += math.Pow(desired[i]-actual[i], 2)
	}
	return 0.5 * sum
}
