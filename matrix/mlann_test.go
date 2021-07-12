package matrix

import (
	"fmt"
	"math/rand"
	"testing"
)

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

func generateData(count int, sd float64) []Sample {
	samples := make([]Sample, 0)
	for i := 0; i < count; i++ {
		in := float64(i) / float64(count)
		samples = append(samples, Sample{
			x: &Matrix{r: 1, c: 1, data: []float64{in}},
			y: &Matrix{r: 1, c: 1, data: []float64{float64(3.0)*in*in + 2 + rand.NormFloat64()*sd}},
		})
	}
	return samples
}

func TestTrain(t *testing.T) {
	//generate the samples
	sd := float64(1.0) / 10.0
	samples := generateData(100, sd)

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

	for idx := 0; idx < 10; idx++ {
		fmt.Println("2.0, 2.75, 5.0")
		fmt.Printf("%.5f, %.5f, %.5f\n", out0.data[0], out1.data[0], out2.data[0])

		err = ann.Train(samples, false, 1000)
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
	if out0.data[0] < 1.9 || out0.data[0] > 2.1 ||
		out1.data[0] < 2.7 || out1.data[0] > 2.8 ||
		out2.data[0] < 4.9 || out2.data[0] > 5.1 {
		t.Error("Results did not fit the provided function accurately")
	}
}

func makeNetwork(layers, hidden, inputs, outputs int, eta, lambda float64) (*MLann, error) {
	ann := NewMLann(eta, lambda)
	input := NewLayer(inputs, hidden, &Linear{}, &LinearPrime{}, true)
	err := ann.AddLayer(input)
	if err != nil {
		return nil, err
	}
	for idx := 0; idx < layers; idx++ {
		layer := NewLayer(hidden, hidden, &Sigmoid{}, &SigmoidPrime{}, false)
		err = ann.AddLayer(layer)
		if err != nil {
			return nil, err
		}
	}

	output := NewLayer(hidden, outputs, &Linear{}, &LinearPrime{}, false)
	err = ann.AddLayer(output)
	if err != nil {
		return nil, err
	}
	return ann, nil
}
func TestSigmoid(t *testing.T) {
	//generate the samples
	sd := float64(1.0) / 20.0
	samples := generateData(100, sd)

	//Initialize the network
	ann, err := makeNetwork(2, 5, 1, 1, 0.8, 0.1)
	if err != nil {
		t.Error(err.Error())
		return
	}
	zero := &Matrix{r: 1, c: 1, data: []float64{0.0}}
	middle := &Matrix{r: 1, c: 1, data: []float64{0.5}}
	end := &Matrix{r: 1, c: 1, data: []float64{1.0}}
	//Train the network
	out0, _ := ann.FeedForward(zero)
	out1, _ := ann.FeedForward(middle)
	out2, _ := ann.FeedForward(end)

	for idx := 0; idx < 20; idx++ {
		fmt.Println("2.0, 2.75, 5.0")
		fmt.Printf("%.8f, %.8f, %.8f\n", out0.data[0], out1.data[0], out2.data[0])

		err = ann.Train(samples, false, 1000)
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
	if out0.data[0] < 1.9 || out0.data[0] > 2.1 ||
		out1.data[0] < 2.7 || out1.data[0] > 2.8 ||
		out2.data[0] < 4.9 || out2.data[0] > 5.1 {
		t.Error("Results did not fit the provided function accurately")
	}
}

func generate2DData(count int, sd float64) []Sample {
	samples := make([]Sample, 0)
	for i := 0; i < count; i++ {
		samples = append(samples, Sample{
			x: &Matrix{r: 2, c: 1, data: []float64{rand.NormFloat64() * sd, rand.NormFloat64() * sd}},
			y: &Matrix{r: 2, c: 1, data: []float64{1.0, 0.0}},
		})
	}
	for i := 0; i < count; i++ {
		samples = append(samples, Sample{
			x: &Matrix{r: 2, c: 1, data: []float64{rand.NormFloat64()*sd + 5.0, rand.NormFloat64()*sd + 5.0}},
			y: &Matrix{r: 2, c: 1, data: []float64{0.0, 1.0}},
		})
	}
	return samples
}
func TestClassification(t *testing.T) {
	ann, err := makeNetwork(2, 8, 2, 2, 1.0, 0.1)
	if err != nil {
		t.Error(err.Error())
		return
	}

	samples := generate2DData(200, 1.0)
	zero := &Matrix{r: 2, c: 1, data: []float64{0.0, 0.0}}
	five := &Matrix{r: 2, c: 1, data: []float64{5.0, 5.0}}
	out0, _ := ann.FeedForward(zero)
	out5, _ := ann.FeedForward(five)
	fmt.Println("out0 expected {1.0,0.0}", out0)
	fmt.Println("out5 expected {0.0,1.0}", out5)

	for idx := 0; idx < 10; idx++ {
		err = ann.Train(samples, false, 1000)
		if err != nil {
			t.Error(err.Error())
			return
		}
		out0, _ = ann.FeedForward(zero)
		out5, _ = ann.FeedForward(five)
		fmt.Println("out0 expected {1.0,0.0}", out0)
		fmt.Println("out5 expected {0.0,1.0}", out5)
	}
}
