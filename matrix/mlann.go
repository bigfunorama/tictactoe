package matrix

import (
	"fmt"
	"math"
)

//MLann is a multi-layer artifical neural network
type MLann struct {
	layers []*Layer
	eta    float64
	lambda float64
}

//NewMLann produces a new MLann with learning rate eta and regularization parameter lambda
func NewMLann(eta, lambda float64) *MLann {
	layers := make([]*Layer, 0)
	return &MLann{layers: layers, eta: eta}
}

//Layers proivdes the number of layers currently installed in the network
func (m *MLann) Layers() int {
	return len(m.layers)
}

func (m *MLann) String() string {
	out := ""
	for idx := 0; idx < m.Layers(); idx++ {
		out = fmt.Sprintf("%s\nlayer %d\n", out, idx)
		out += m.layers[idx].String()
	}
	return out
}

//IncompatibleLayerError indicates that the inputs of the layer to be added are
//incompatible with the outputs of the prior layer already in the network.
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

//NewSample constructs a Sample and returns it
func NewSample(x, y *Matrix) Sample {
	return Sample{x: x, y: y}
}

func (s Sample) X() *Matrix {
	return s.x
}

func (s Sample) Y() *Matrix {
	return s.y
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

//Train iterates over the samples provide training the network to minize the
//error given the training samples.
func (m *MLann) Train(samples []Sample, verbose bool, iterations int) error {
	for idx := 0; idx < iterations; idx++ {
		err := m.iterate(samples, verbose)
		if err != nil {
			return err
		}
	}
	return nil
}

//iterate applies the error produced by the network for the given input
//sample x to the current network.
func (m *MLann) iterate(samples []Sample, verbose bool) error {
	updatesW := make([]*Matrix, 0)
	updatesB := make([]*Matrix, 0)
	for idx := 0; idx < len(m.layers); idx++ {
		updatesW = append(updatesW, NewMatrix(m.layers[idx].weights.r, m.layers[idx].weights.c))
		updatesB = append(updatesB, NewMatrix(m.layers[idx].bias.r, m.layers[idx].bias.c))
	}

	for _, sample := range samples {
		nablaw, nablab, err := m.backprop(sample, len(samples), false)
		if err != nil {
			return err
		}
		for idx := 0; idx < len(m.layers); idx++ {
			updatesW[idx], _ = updatesW[idx].Add(nablaw[idx])
			updatesB[idx], _ = updatesB[idx].Add(nablab[idx])
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
		if verbose {
			fmt.Println("weights", idx)
			fmt.Println(updateW)
		}
		updateB := updatesB[idx].ScalarMul(m.eta / float64(len(samples)))
		updateB, _ = m.layers[idx].bias.Sub(updateB)
		m.layers[idx].bias = updateB
	}
	return nil
}

//backprop produces the direction of update for this sample propagated through the network
func (m *MLann) backprop(in Sample, batchSize int, verbose bool) ([]*Matrix, []*Matrix, error) {
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
