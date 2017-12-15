package statistics

import (
	"math"
	"sort"
)

//Sum collects all numbers in an array and returns their sum
func Sum(data []float64) float64 {
	sum := 0.0
	for _, d := range data {
		sum += d
	}
	return sum
}

//Mean calculates the average over a set of values
func Mean(data []float64) float64 {
	return Sum(data) / float64(len(data))
}

//Std calculates the standard deviation of the data
func Std(data []float64) float64 {
	mean := Mean(data)
	sum := 0.0
	for _, d := range data {
		sum += math.Pow(d-mean, 2)
	}
	return math.Sqrt(sum * (1 / float64(len(data))))
}

//Frequency creates a map of each value present in the array and the subsiquent count
func Frequency(data []float64) map[float64]int {
	freq := make(map[float64]int)
	for _, d := range data {
		freq[d]++
	}
	return freq
}

//Mode calculates the value(s) which occur most frequently in the array
func Mode(data []float64) []float64 {
	freq := Frequency(data)
	criterion := 0
	var outputs []float64

	for k, v := range freq {
		if v > criterion {
			criterion = v
			outputs = []float64{k}
		} else if v == criterion {
			outputs = append(outputs, k)
		}
	}

	return outputs
}

//Variance returns the standard deviation squared of the data
func Variance(data []float64) float64 {
	return math.Pow(Std(data), 2)
}

//Min returns the smallest value found in the data
func Min(data []float64) float64 {
	sort.Float64s(data)
	return data[0]
}

//Max returns the largest value in the data
func Max(data []float64) float64 {
	sort.Float64s(data)
	return data[len(data)-1]
}

//Median calculates the 50th precentile of the data
func Median(data []float64) float64 {
	return Percentile(data, 50)[0]
}

//Percentile calculates the precentile values for the specified precentages of the data
func Percentile(data []float64, prec ...int) []float64 {
	sort.Float64s(data)
	var out []float64

	incrementAmount := 100 / len(data)
	currentIndex := 0
	for i := 0; i < 100; i += incrementAmount {
		for _, v := range prec {
			if v >= i && v <= i+incrementAmount {
				out = append(out, data[currentIndex])
			}
		}
		currentIndex++
	}
	return out
}

//IQR calculates the inter-quartile range of the dataset
func IQR(data []float64) float64 {
	p := Percentile(data, 75, 25)
	return p[0] - p[1]
}

//Range calculates the range of the dataset
func Range(data []float64) float64 {
	return Max(data) - Min(data)
}
