package matrix

import (
	"fmt"
	"math"
	"math/rand"
)

//Layer provides a single layer in ANN encapsulating weights and
//an activation function.
type Layer struct {
	//Activation function to be applied
	h       Activation
	hprime  Activation
	weights *Matrix
	bias    *Matrix
}

//NewLayer produces a Layer with inputs number of inputs, outputs number of outputs
//with activation function h. The layer will be randomly initialized and produce the
//output h(w * a + b) where h is the activation function w is the weight matrix with
//dimensionality inputs x outputs and b is a bias vector of length outputs. If the
//initilLayer flag is set, then the bias terms will be initialized to 0.
func NewLayer(inputs, outputs int, h, hprime Activation, initialLayer bool) *Layer {
	//set the intial weights as 0 mean Gaussian and sd of 1/sqrt(inputs)
	//unless this is the initial layer where we set sd to 1.0
	sd := math.Sqrt(float64(2.0) / float64(inputs+outputs))
	if initialLayer {
		sd = float64(1.0)
	}

	output := make([]float64, inputs*outputs)
	for i := 0; i < len(output); i++ {
		output[i] = rand.NormFloat64() * sd
	}

	bias := make([]float64, outputs)
	if !initialLayer {
		for i := 0; i < len(bias); i++ {
			bias[i] = 0.1
		}
	}

	return &Layer{
		h:       h,
		hprime:  hprime,
		weights: &Matrix{r: outputs, c: inputs, data: output},
		bias:    &Matrix{r: outputs, c: 1, data: bias},
	}
}

//Outputs provides the count of the outputs produced by this layer
func (l *Layer) Outputs() int {
	return l.bias.r
}

//Inputs provides the count of the inputs required by this layer
func (l *Layer) Inputs() int {
	return l.weights.c
}

//Activate returns l.h(l.w * x + l.b)
func (l *Layer) Activate(x *Matrix) (*Matrix, error) {
	tmp, err := l.weights.Mul(x)
	if err != nil {
		return nil, err
	}
	tmp, err = tmp.Add(l.bias)
	if err != nil {
		return nil, err
	}
	return tmp.Apply(l.h), nil
}

//String produces a string representation of the layer using 5 places of precision
// [ weights ] [ bias ]
func (l *Layer) String() string {
	out := ""
	for i := 0; i < l.weights.r; i++ {
		out += "["
		for j := 0; j < l.weights.c; j++ {
			out = fmt.Sprintf("%s %.5f", out, l.weights.data[i*l.weights.c+j])
		}
		out += " ]    [ "
		for j := 0; j < l.bias.c; j++ {
			out = fmt.Sprintf("%s %.5f", out, l.bias.data[i*l.bias.c+j])
		}
		out += " ]\n"
	}
	return out
}
