package nn

import "math"

func GenerateExample(in, out []float64) Example {
	return Example{
		Input:  in,
		Output: out,
	}
}

func SquaredError(desired, actual []float64) float64 {
	if len(desired) != len(actual) {
		panic("Size of Desired and Actual Arrays are Not Equal.")
	}

	var sum float64 = 0
	for i := range desired { sum += math.Pow(desired[i] - actual[i], 2) }
	return 0.5 * sum
}