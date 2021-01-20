package matrix

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestLayer(t *testing.T) {
	layer := NewLayer(2, 10, &RELU{}, &RELUPrime{}, true)

	if layer.Inputs() != 2 {
		t.Errorf("Expected 2 inputs but got %d", layer.Inputs())
	}

	if layer.Outputs() != 10 {
		t.Errorf("Expected 10 outputs but got %d", layer.Outputs())
	}

	if layer.weights.c != 2 {
		t.Errorf("Expected two columns in the weights matrix but got %d", layer.weights.c)
	}

	if layer.weights.r != 10 {
		t.Errorf("Expected ten rows in the weights matrix but got %d", layer.weights.r)
	}

	for idx := 0; idx < layer.Outputs(); idx++ {
		if layer.bias.data[idx] != 0.0 {
			t.Errorf("expected 0.0 for bias on an input layer, got %.5f", layer.bias.data[idx])
		}
	}
	x := &Matrix{r: 2, c: 1, data: []float64{1.0, 1.0}}
	out, err := layer.Activate(x)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	fmt.Println(out)
	fmt.Println(layer)
}

func TestAddLayer(t *testing.T) {
	ann := NewMLann(0.05, 0.1)
	layer := NewLayer(2, 10, &RELU{}, &RELUPrime{}, true)
	err := ann.AddLayer(layer)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	layer2 := NewLayer(10, 2, &RELU{}, &RELUPrime{}, false)
	err = ann.AddLayer(layer2)
	if err != nil {
		t.Errorf(err.Error())
	}
}

func TestFeedForward(t *testing.T) {
	ann := NewMLann(0.05, 0.1)
	layer := NewLayer(2, 10, &RELU{}, &RELUPrime{}, true)
	err := ann.AddLayer(layer)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	layer2 := NewLayer(10, 2, &RELU{}, &RELUPrime{}, false)
	err = ann.AddLayer(layer2)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	input := NewMatrix(2, 1)
	input.data[0] = 0.1
	input.data[1] = 0.3
	output, err := ann.FeedForward(input)
	if err != nil {
		t.Error(err.Error())
		return
	}
	fmt.Println(output)
}

func TestDistance(t *testing.T) {
	lt := &Matrix{r: 3, c: 1, data: []float64{2.0, 2.0, 2.0}}
	rt := &Matrix{r: 3, c: 1, data: []float64{4.0, 4.0, 4.0}}
	out, err := distance(lt, rt)
	if err != nil {
		t.Error(err.Error())
	}
	if out != 12.0 {
		t.Errorf("expected 12.0 got %.5f", out)
	}
}

func TestTrain(t *testing.T) {
	//generate the samples
	sd := float64(1.0) / 10.0
	samples := make([]Sample, 0)
	for i := 0; i < 100; i++ {
		in := float64(i) / 100.0
		samples = append(samples, Sample{
			x: &Matrix{r: 1, c: 1, data: []float64{in}},
			y: &Matrix{r: 1, c: 1, data: []float64{float64(3.0)*in*in + 2 + rand.NormFloat64()*sd}},
		})
	}

	//Initialize the network
	ann := NewMLann(0.5, 0.)
	layer := NewLayer(1, 8, &Linear{}, &LinearPrime{}, true)
	err := ann.AddLayer(layer)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	layer2 := NewLayer(8, 8, &Sigmoid{}, &SigmoidPrime{}, false)
	err = ann.AddLayer(layer2)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	layer3 := NewLayer(8, 8, &Sigmoid{}, &SigmoidPrime{}, false)
	err = ann.AddLayer(layer3)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	layer4 := NewLayer(8, 1, &Linear{}, &LinearPrime{}, false)
	err = ann.AddLayer(layer4)
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	zero := &Matrix{r: 1, c: 1, data: []float64{0.0}}
	middle := &Matrix{r: 1, c: 1, data: []float64{0.5}}
	end := &Matrix{r: 1, c: 1, data: []float64{1.0}}
	//Train the network
	out0, _ := ann.FeedForward(zero)
	out1, _ := ann.FeedForward(middle)
	out2, _ := ann.FeedForward(end)

	for idx := 0; idx < 10000; idx++ {
		fmt.Println("2.0, 2.75, 5.0")
		fmt.Printf("%.5f, %.5f, %.5f\n", out0.data[0], out1.data[0], out2.data[0])

		err = ann.Train(samples, false)
		if err != nil {
			t.Error(err.Error())
			return
		}
		dist, err2 := ann.SquaredError(samples)
		if err2 != nil {
			t.Error(err2.Error())
			return
		}
		fmt.Println(idx, dist)
		out0, _ = ann.FeedForward(zero)
		out1, _ = ann.FeedForward(middle)
		out2, _ = ann.FeedForward(end)

	}
	for idx := 0; idx < len(ann.layers); idx++ {
		fmt.Println("layer", idx)
		fmt.Println(ann.layers[idx])
	}

	fmt.Println("2.0, 2.75, 5.0")
	fmt.Printf("%.5f, %.5f, %.5f\n", out0.data[0], out1.data[0], out2.data[0])
}
