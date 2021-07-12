package matrix

import (
	"fmt"
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
