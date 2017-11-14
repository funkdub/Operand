package main

import (
	"fmt"
	"./nn"
)

func main() {

	fmt.Println("Starting Neural Network...")

	network := nn.InitializeRandomNetwork([]int{5, 3, 2, 4}, 0.05, 10)

	ex := nn.Example{
		Input:  []float64{5, 4, 3, 2, 1},
		Output: []float64{1, 0, 0, 0},
	}

	fmt.Println(network.ForwardPropagateInput(ex.Input))

	fmt.Println(network.BackPropegateExample(ex))







}
