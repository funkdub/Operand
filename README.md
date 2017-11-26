# Operand

![Go Report Card](https://goreportcard.com/badge/github.com/MorganGallant/Operand)

Operand is a deep artificial neural network (ANN) library for Google's Go, used to easily create deep feed-forward neural networks, with a dynamic hidden structure. The goal of the project is to simplify the creation of neural networks for numerical pattern recognition, for varying uses.



## Installation
```
go get github.com/MorganGallant/Operand
```

Some IDE software may do this step for you (such as GoLand). This command will download the library locally into your project.

## Usage

First, import the library into your Go package.
```
import "github.com/MorganGallant/Operand"
```

To create a new network, and specify parameters, use the ```InitializeNewNetwork``` function. It returns a network instance.

```
network := Operand.nn.InitializeNewNetwork(Structure, Learning Rate, MomentumFactor, Epochs, Activation Func, Activation Func Derivative)
```

The structure of the network is typed as  ```[]int```. It must have at minimum 3 layers: 1 input, n hiddens, 1 output.
The learning rate and momentum factor are both type ```float64```, typically between 0 and 1. The epoch count is an ```int```. Both the activation function and it's derivative are of type ```func(float64) float64```.

Example of Structure (MNIST with two hidden layers): ```[]int{784, 10, 8, 10}```.

To train the network, you must first generate examples of type ```Operand.nn.Example```, which takes in two ```[]float64``` arrays, one for the input of the network, the other as the desired output. You can use the ```Operand.nn.GenerateExample (in, out []float64) Example``` function to assist in the creation of example sets.

Once the example set is created, train the network using the ```network.TrainNetwork(Example Set, Debug Mode)```, where the example set is of type ```[]Operand.nn.Example``` and the debug mode is specified by a ```bool``` value. Ensure that the input/output size of the network matches the size of the example input and output.

To evaluate the network on a test/hold-out set, use the ```EvaluateNetwork``` function, which returns an accuracy percentage as a ```float64``` value.

```
accuracy := network.EvaluateNetwork(Example Set)
```

The example set is of type ```[]Operand.nn.Example```.

To forward propagate a ```[]float64``` through the network, and get the output ```[]float64```, use the ```ForwardPropagateInput``` function.

```
output := network.ForwardPropagateInput(Input)
```

The input is of type ```[]float64```, and it's size much match the size of the input layer of the network.
The function will return a ```[]float64``` with size equal to the output layer of the network.

## Activation Functions

As part of Operand, there are some activation functions to choose from.  For now, these are stored in the ```Operand.nn``` package, however it is my goal to expand this library to contain a math-heavy package with matrix and statistical operations supported.

When creating a new network, you must specify which functions to use, or create your own.

Sigmoid:
```
Operand.nn.Sigmoid(float64) float64
Operand.nn.DSigmoid(float64) float64
```

ReLU:
```
Operand.nn.Relu(float64) float64
Operand.nn.DRelu(float64) float64
```

Hyperbolic Tangent:
```
Operand.nn.TanH(float64) float64
Operand.nn.DTanH(float64) float64
```

## Example

The following example will showcase the library in action, approximating the sin function between 0-1 with 99.98% accuracy using a four layer fully connected neural network. The learning rate and momentum factor was set to 0.2. 100 Epochs were completed during training.

Source Code:
```
//Initialize new arrays to store example sets (train and test)
var examplesTrain, examplesTest []Operand.nn.Example

//Create the network, with input size of one and output size of one.
network := Operand.nn.InitializeNewNetwork([]int{1, 10, 8, 1}, 0.2, 0.2, 100, Operand.nn.Sigmoid, Operand.nn.DSigmoid)

//Generate the test and training data set, in this case, to have the network approximate the sin function.
for i := 0; i < 10000; i++ {
    valTrain := rand.Float64()
    valTest := rand.Float64()

    examplesTrain = append(examplesTrain, Operand.nn.GenerateExample([]float64{valTrain}, []float64{math.Sin(valTrain)}))
    examplesTest = append(examplesTest, Operand.nn.GenerateExample([]float64{valTest}, []float64{math.Sin(valTest)}))
}

//Train the network on the training set. You may turn debug off if needed.
network.TrainNetwork(examplesTrain, true)

//Evaluate the network on the test set, and print the result to the console.
fmt.Println("Final Evaluation:", network.EvaluateNetwork(examplesTest), "%")
```

Output of the Program:
```
...
Epoch 96 . Accuracy Precentage Achieved: 99.98408073977802
Epoch 97 . Accuracy Precentage Achieved: 99.98410529209896
Epoch 98 . Accuracy Precentage Achieved: 99.9841297896758
Epoch 99 . Accuracy Precentage Achieved: 99.98415423412914
Final Evaluation: 99.98387637656883 %
```



