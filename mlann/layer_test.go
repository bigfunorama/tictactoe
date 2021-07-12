package mlann

import (
	"fmt"
	"strings"
	"testing"
)

func TestLayer(t *testing.T) {
	layer, err := NewLayer(2, 10, "RELU", "RELUPrime", true)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
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
	layer, err := NewLayer(2, 10, "RELU", "RELUPrime", true)
	if err != nil {
		t.Error(err.Error())
		return
	}
	err = ann.AddLayer(layer)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	layer2, err := NewLayer(10, 2, "RELU", "RELUPrime", false)
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	err = ann.AddLayer(layer2)
	if err != nil {
		t.Errorf(err.Error())
	}
}

func TestLoadLayer(t *testing.T) {
	weights, err := LoadMatrix(strings.NewReader("2,2,1.0,2.0,3.0,4.0"))
	if err != nil {
		t.Error(err.Error())
	}
	bias, err := LoadMatrix(strings.NewReader("2,1,1.0,0.0"))
	if err != nil {
		t.Error(err.Error())
	}
	h, err := getActivation("Linear")
	if err != nil {
		t.Error(err.Error())
	}
	hprime, err := getActivation("LinearPrime")
	if err != nil {
		t.Error(err.Error())
	}
	layer := &Layer{h: h, hprime: hprime, weights: weights, bias: bias}

	bb := &strings.Builder{}
	layer.Write(bb)
	fmt.Println(bb.String())
	layer2, err := LoadLayer(strings.NewReader(bb.String()))
	if err != nil {
		t.Error(err.Error())
	}
	if !layer2.weights.Equals(layer.weights) {
		t.Error("weight matrices are not equal")
	}

	if !layer2.bias.Equals(layer.bias) {
		t.Error("bias matrices are not equal")
	}

	if layer2.h.Name() != layer.h.Name() {
		t.Errorf("expected %s, got %s", layer.h.Name(), layer2.h.Name())
	}

	if layer2.hprime.Name() != layer.hprime.Name() {
		t.Errorf("expected %s, got %s", layer.hprime.Name(), layer2.hprime.Name())
	}
}
