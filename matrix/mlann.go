package matrix

import (
	"fmt"
	"math"
	"math/rand"
)

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
		return float64(0.0)
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

//Layer provides a single layer in ANN encapsulating weights and
//an activation function.
type Layer struct {
	//Activation function to be applied
	h       Activation
	weights *Matrix
	bias    *Matrix
}

//NewLayer produces a Layer with inputs number of inputs, outputs number of outputs
//with activation function h. The layer will be randomly initialized and produce the
//output h(w * a + b) where h is the activation function w is the weight matrix with
//dimensionality inputs x outputs and b is a bias vector of length outputs. If the
//initilLayer flag is set, then the bias terms will be initialized to 0.
func NewLayer(inputs, outputs int, h Activation, initialLayer bool) *Layer {
	sd := math.Sqrt(float64(1.0) / float64(inputs))
	//set the intial weights as 0 mean Gaussian and sd of 1/sqrt(inputs)
	output := make([]float64, inputs*outputs)
	for i := 0; i < len(output); i++ {
		output[i] = rand.NormFloat64() * sd
	}
	//set the initial bias as 0 mean Gaussian and sd of 1
	bias := make([]float64, outputs)
	if !initialLayer {
		for i := 0; i < len(bias); i++ {
			bias[i] = rand.NormFloat64()
		}
	}

	return &Layer{
		h:       h,
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

func (m *Layer) String() string {
	out := ""
	for i := 0; i < m.weights.r; i++ {
		out += "["
		for j := 0; j < m.weights.c; j++ {
			out = fmt.Sprintf("%s %.5f", out, m.weights.data[i*m.weights.c+j])
		}
		out += " ]    [ "
		for j := 0; j < m.bias.c; j++ {
			out = fmt.Sprintf("%s %.5f", out, m.bias.data[i*m.bias.c+j])
		}
		out += " ]\n"
	}
	return out
}

//MLann is a multi-layer artifical neural network
type MLann struct {
	layers []*Layer
	eta    float64
}

func NewMLann(eta float64) *MLann {
	layers := make([]*Layer, 0)
	return &MLann{layers: layers, eta: eta}
}

type IncompatibleLayerError struct{ val string }

func (ile *IncompatibleLayerError) Error() string {
	return "Incompatible layer provided"
}

//AddLayer appends the provided layer to the existing network
//The inputs required by the added layer must be equal to the
//number of outputs of the previous layer.
func (m *MLann) AddLayer(layer *Layer) error {
	if len(m.layers) == 0 {
		m.layers = append(m.layers, layer)
		return nil
	}

	if m.layers[len(m.layers)-1].Outputs() != layer.Inputs() {
		return &IncompatibleLayerError{val: fmt.Sprintf("current number of outputs is %d, but layer requires inputs %d", m.layers[len(m.layers)-1].Outputs(), layer.Inputs())}
	}
	m.layers = append(m.layers, layer)
	return nil
}

//FeedForward takes an input matrix and applies the MLP layers to it
//to produce the current output from the network
func (m *MLann) FeedForward(input *Matrix) (*Matrix, error) {
	tmp := input.Clone()
	err := error(nil)
	for idx := 0; idx < len(m.layers); idx++ {
		tmp, err = m.layers[idx].Activate(tmp)
		if err != nil {
			return nil, err
		}
	}
	return tmp, nil
}

//Sample pairs an input x with an expected output y
type Sample struct {
	x *Matrix
	y *Matrix
}

//Train applies the error produced by the network for the given input
//sample x to the current network.
func (m *MLann) Train(samples []Sample) error {
	for _, sample := range samples {
		err := m.Update(sample, len(samples))
		if err != nil {
			return err
		}
	}
	return nil
}

//Update uses the x,y pair to update weights in the network based on this sample
func (m *MLann) Update(in Sample, batchSize int) error {
	zs := make([]*Matrix, len(m.layers))
	as := make([]*Matrix, len(m.layers))
	//go forward through the network to produce the outcomes at the various layers
	a := in.x
	for idx := 0; idx < len(m.layers); idx++ {
		if idx > 0 {
			a = as[idx-1]
		}
		tmp, err := m.layers[idx].weights.Mul(a)
		if err != nil {
			fmt.Println("Mul", m.layers[idx].weights.r, " x ", m.layers[idx].weights.c, " * ", a.r, " x ", a.c)
			fmt.Println("Mul", idx, m.layers[idx].weights, a)
			return err
		}
		tmp, err = tmp.Add(m.layers[idx].bias)
		if err != nil {
			fmt.Println("Add", idx, tmp, m.layers[idx].bias)
			return err
		}
		zs[idx] = tmp
		as[idx] = tmp.Apply(m.layers[idx].h)
	}

	//then go back and compute the gradients
	nablab := make([]*Matrix, len(m.layers))
	nablaw := make([]*Matrix, len(m.layers))
	L := len(as) - 1
	delta, err := as[L].Sub(in.y)
	sp := &RELUPrime{}
	if err != nil {
		fmt.Println("as[len(as)-1].Sub(in.y)", L, as[L], in.y)
		return err
	}
	delta, err = delta.Hadamard(zs[L].Apply(sp))
	if err != nil {
		fmt.Println("delta.Hadamard(zs[len(zs)-1].Apply(sp))", delta, zs[L])
		return err
	}
	nablab[L] = delta
	tmp, err := delta.Mul(as[L-1].Transpose())
	if err != nil {
		fmt.Println("delta.Mul(as[len(as)-1].Transpose())\n", delta, as[len(as)-1], "transpose")
		return err
	}
	nablaw[L] = tmp
	for idx := L - 1; idx >= 0; idx-- {

		//Compute the next delta
		delta, err = m.layers[idx+1].weights.Transpose().Mul(delta)
		if err != nil {
			fmt.Println("m.layers[idx].weights.Transpose().Mul(delta)\n", m.layers[idx].weights.Transpose(), delta)
			return err
		}
		delta, err = delta.Hadamard(zs[idx].Apply(sp))
		if err != nil {
			fmt.Println("delta.Hadamard(zs[idx].Apply(sp))\n", delta, zs[idx])
			return err
		}

		nablab[idx] = delta
		if idx > 0 {
			tmp, err := delta.Mul(as[idx-1].Transpose())
			if err != nil {
				fmt.Println("delta.Mul(as[idx-1].Transpose())\n", delta, as[idx-1].Transpose())
				return err
			}
			nablaw[idx] = tmp
		} else {
			tmp, err := delta.Mul(in.x.Transpose())
			if err != nil {
				fmt.Println("delta.Mul(as[idx-1].Transpose())\n", delta, as[idx-1].Transpose())
				return err
			}
			nablaw[idx] = tmp
		}
	}

	//then apply the updates through stochastic gradient descent
	for idx := 0; idx < len(m.layers); idx++ {
		updateW := nablaw[idx].ScalarMul(m.eta / float64(batchSize))
		updateW, err = m.layers[idx].weights.Sub(updateW)
		if err != nil {
			fmt.Println("m.layers[idx].weights.Sub(updateW)\n", m.layers[idx].weights, updateW)
			return err
		}
		m.layers[idx].weights = updateW
		updateB := nablab[idx].ScalarMul(m.eta / float64(batchSize))
		updateB, err = m.layers[idx].bias.Sub(updateB)
		if err != nil {
			fmt.Println("m.layers[idx].bias.Sub(updateB)\n", m.layers[idx].bias, updateB)
			return err
		}
		m.layers[idx].bias = updateB
	}
	return nil
}
