package matrix

import "math"

//Activation defines an activation function for a layer in ANN.
type Activation interface {
	Activate(input float64) float64
}

//RELU is a rectified linear unit where any input less than or equal to 0
// produces an output of 0.0 and a linear output otherwise
type RELU struct{}

//Activate applies the activation function to the input and returns the result
func (r *RELU) Activate(input float64) float64 {
	if input < 0 {
		return float64(0.01)
	}
	return input
}

//RELUPrime is the first derivative of the RELU activation function
type RELUPrime struct{}

//Activate computes the value of the derivative at input and returns the result
func (r *RELUPrime) Activate(input float64) float64 {
	if input <= 0.0 {
		return float64(0.0)
	}
	return 1.0
}

//Linear Activation function
type Linear struct{}

//Activate produces an output equal to it's input
func (l *Linear) Activate(input float64) float64 {
	return input
}

//LinearPrime produces the derivitive of the Linear activation function
//so 1.0 regardless of the input
type LinearPrime struct{}

//Activate produces 1.0 as the output regardless of the input.
func (lp *LinearPrime) Activate(input float64) float64 {
	return float64(1.0)
}

//Sigmoid produces the sigmoid function output
type Sigmoid struct{}

//Activate produces 1 / (1 + Exp(-input))
func (s *Sigmoid) Activate(input float64) float64 {
	return float64(1.0) / (1.0 + math.Exp(-input))
}

//SigmoidPrime produces the first derivitive output of the sigmoid function
type SigmoidPrime struct{}

//Activate produces Sigmoid(x) * ( 1 - Sigmoid(x))
func (sp *SigmoidPrime) Activate(input float64) float64 {
	s := float64(1.0) / (1.0 + math.Exp(-input))
	return s * (float64(1.0) - s)
}
