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

type Linear struct{}

func (l *Linear) Activate(input float64) float64 {
	return input
}

type LinearPrime struct{}

func (lp *LinearPrime) Activate(input float64) float64 {
	return float64(1.0)
}

type Sigmoid struct{}

func (s *Sigmoid) Activate(input float64) float64 {
	return float64(1.0) / (1.0 + math.Exp(-input))
}

type SigmoidPrime struct{}

func (sp *SigmoidPrime) Activate(input float64) float64 {
	s := float64(1.0) / (1.0 + math.Exp(-input))
	return s * (float64(1.0) - s)
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
	sd := math.Sqrt(float64(1.0) / float64(inputs))
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
	lambda float64
}

func NewMLann(eta, lambda float64) *MLann {
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

func (s Sample) String() string {
	return fmt.Sprintf("%s\n%s", s.x, s.y)
}

func distance(lt, rt *Matrix) (float64, error) {
	if lt.c != 1 || rt.c != 1 || lt.r != rt.r {
		return 0.0, &IncompatibleMatrixError{}
	}
	dist := float64(0)
	for idx := 0; idx < lt.r; idx++ {
		dist += ((lt.data[idx] - rt.data[idx]) * (lt.data[idx] - rt.data[idx]))
	}
	return math.Sqrt(dist), nil
}

//SquaredError returns the sum of squares error for the given slice of samples and
//the current network.
func (m *MLann) SquaredError(samples []Sample) (float64, error) {
	se := 0.0
	for _, sample := range samples {
		out, err := m.FeedForward(sample.x)
		if err != nil {
			return -1.0, err
		}
		dist, err := distance(out, sample.y)
		if err != nil {
			return -1.0, err
		}
		se += dist
	}
	return se, nil
}

//Train applies the error produced by the network for the given input
//sample x to the current network.
func (m *MLann) Train(samples []Sample, verbose bool) error {
	updatesW := make([]*Matrix, 0)
	updatesB := make([]*Matrix, 0)
	for idx := 0; idx < len(m.layers); idx++ {
		updatesW = append(updatesW, NewMatrix(m.layers[idx].weights.r, m.layers[idx].weights.c))
		updatesB = append(updatesB, NewMatrix(m.layers[idx].bias.r, m.layers[idx].bias.c))
	}

	for s1, sample := range samples {
		nablaw, nablab, err := m.Update(sample, len(samples), verbose)
		if err != nil {
			return err
		}
		for idx := 0; idx < len(m.layers); idx++ {
			updatesW[idx], _ = updatesW[idx].Add(nablaw[idx])
			updatesB[idx], _ = updatesB[idx].Add(nablab[idx])
			if s1 == 0 && verbose {
				fmt.Println("nablab", idx, nablab[idx])
				fmt.Println("nablaw", idx, nablaw[idx])
			}
		}
	}
	//then apply the updates through stochastic gradient descent
	for idx := 0; idx < len(m.layers); idx++ {
		updateW := updatesW[idx].ScalarMul(m.eta / float64(len(samples)))
		if verbose {
			fmt.Println("updateW", idx)
			fmt.Println(updateW)
		}

		tmp := m.layers[idx].weights.ScalarMul(float64(1.0) - m.eta*m.lambda/float64(len(samples)))
		updateW, _ = tmp.Sub(updateW)
		m.layers[idx].weights = updateW

		updateB := updatesB[idx].ScalarMul(m.eta / float64(len(samples)))
		updateB, _ = m.layers[idx].bias.Sub(updateB)
		m.layers[idx].bias = updateB
	}
	return nil
}

//Compute the direction of update for this sample
func (m *MLann) Update(in Sample, batchSize int, verbose bool) ([]*Matrix, []*Matrix, error) {
	aks := make([]*Matrix, len(m.layers))
	hks := make([]*Matrix, len(m.layers))
	//go forward through the network to produce the outcomes at the various layers
	h := in.x
	for k := 0; k < len(m.layers); k++ {
		if k > 0 {
			h = hks[k-1]
		}
		hk, err := m.layers[k].weights.Mul(h)
		if err != nil {
			fmt.Println("Mul", m.layers[k].weights.r, " x ", m.layers[k].weights.c, " * ", h.r, " x ", h.c)
			fmt.Println("Mul", k, m.layers[k].weights, h)
			return nil, nil, err
		}
		hk, err = hk.Add(m.layers[k].bias)
		if err != nil {
			fmt.Println("Add", k, hk, m.layers[k].bias)
			return nil, nil, err
		}
		aks[k] = hk
		hks[k] = hk.Apply(m.layers[k].h)
	}

	//then go back and compute the gradients
	nablab := make([]*Matrix, len(m.layers))
	nablaw := make([]*Matrix, len(m.layers))
	L := len(hks) - 1
	delta, err := hks[L].Sub(in.y)
	if err != nil {
		fmt.Println("as[len(as)-1].Sub(in.y)", L, hks[L], in.y)
		return nil, nil, err
	}

	if verbose {
		fmt.Println("***********************")
		fmt.Println(len(m.layers)-1, "delta = hks[L].Sub(in.y)\n", delta)
		fmt.Println("***********************")
	}

	//Linear output layer
	delta, err = delta.Hadamard(aks[L].Apply(m.layers[L].hprime))
	if err != nil {
		fmt.Println("delta.Hadamard(aks[L].Apply(m.layers[L].hprime))", delta, aks[L])
		return nil, nil, err
	}

	if verbose {
		fmt.Println(len(m.layers)-1, "aks[L]", aks[L])
		fmt.Println(len(m.layers)-1, "aks[L].Apply(m.layers[L].hprime)\n", aks[L].Apply(m.layers[L].hprime))
		fmt.Println(len(m.layers)-1, "delta = delta.Hadamard(aks[L].Apply(m.layers[L].hprime))\n", delta)
		fmt.Println("***********************")
	}

	nablab[L] = delta
	tmp, err := delta.Mul(hks[L-1].Transpose())
	if err != nil {
		fmt.Println("delta.Mul(as[len(as)-1].Transpose())\n", delta, hks[len(hks)-1], "transpose")
		return nil, nil, err
	}
	nablaw[L] = tmp
	if verbose {
		fmt.Println(len(m.layers)-1, "delta = delta.Mul(hks[L-1].Transpose())\n", delta)
		fmt.Println("***********************")
	}

	for k := L - 1; k >= 0; k-- {

		//Compute the next delta
		delta, err = m.layers[k+1].weights.Transpose().Mul(delta)
		if err != nil {
			fmt.Println("m.layers[idx].weights.Transpose().Mul(delta)\n", m.layers[k].weights.Transpose(), delta)
			return nil, nil, err
		}

		if verbose {
			fmt.Println(k+1, "m.layers[k+1].weights.Transpose()\n", m.layers[k+1].weights.Transpose())
			fmt.Println(k, "delta = m.layers[k+1].weights.Transpose().Mul(delta)\n", delta)
			fmt.Println("***********************")
		}

		delta, err = delta.Hadamard(aks[k].Apply(m.layers[k].hprime))
		if err != nil {
			fmt.Println("delta.Hadamard(zs[idx].Apply(sp))\n", delta, aks[k])
			return nil, nil, err
		}
		if verbose {
			fmt.Println(k, "aks[k]", aks[k])
			fmt.Println(k, "aks[k].Apply(m.layers[k].hprime)\n", aks[k].Apply(m.layers[k].hprime))
			fmt.Println(k, "delta = delta.Hadamard(aks[k].Apply(m.layers[k].hprime))\n", delta)
			fmt.Println("***********************")
		}

		nablab[k] = delta
		var tmp *Matrix
		if k > 0 {
			tmp = hks[k-1].Transpose()
		} else {
			tmp = in.x.Transpose()
		}
		out, err := delta.Mul(tmp)
		if err != nil {
			return nil, nil, err
		}
		nablaw[k] = out
	}
	return nablaw, nablab, nil
}
