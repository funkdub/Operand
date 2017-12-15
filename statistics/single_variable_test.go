package statistics

import (
	"math"
	"testing"
)

func TestMean(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	if mean := Mean(data); mean != 3 {
		t.Error("Mean is Incorrect. Desired: 3 Actual:", mean)
	}
}

func TestStd(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	if std := Std(data); std != math.Sqrt(2) {
		t.Error("Standard Deviation is Incorrect. Desired: 1.414 Actual:", std)
	}
}

func TestFrequency(t *testing.T) {
	data := []float64{1, 2, 2, 2, 3, 3, 4, 5}
	exOut := make(map[float64]int)
	exOut[5] = 1
	exOut[1] = 1
	exOut[2] = 3
	exOut[3] = 2
	exOut[4] = 1
	freq := Frequency(data)
	for k, v := range freq {
		if exOut[k] != v {
			t.Error("Frequency is Incorrect. Desired:", k, v, "Actual:", k, freq[k])
		}
	}
}

func TestMode(t *testing.T) {
	data := []float64{1, 2, 2, 2, 3, 3, 4, 5}

	if mode := Mode(data); mode[0] != 2 {
		t.Error("Mode is Incorrect. Desired: 2 Actual:", mode[0])
	}
}

func TestPercentile(t *testing.T) {
	data := []float64{1, 10, 11, 15, 19}

	if prec := Percentile(data, 50)[0]; prec != 11 {
		t.Error("Precentile is Incorrect. Desired:", 11, "Actual:", prec)
	}
}

func TestMedian(t *testing.T) {
	data := []float64{1, 10, 11, 15, 19}
	if median := Median(data); median != 11 {
		t.Error("Median is Incorrect. Desired:", 11, "Actual:", median)
	}
}
