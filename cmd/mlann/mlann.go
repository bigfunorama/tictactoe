package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/bigfunorama/tictactoe/matrix"
	ui "github.com/gizak/termui/v3"
	"github.com/gizak/termui/v3/widgets"
)

func generateData(count int, sd float64) []matrix.Sample {
	samples := make([]matrix.Sample, 0)
	for i := 0; i < count; i++ {
		in := float64(i) / float64(count)

		x := matrix.NewMatrix(1, 1)
		x.Set(0, 0, in)

		y := matrix.NewMatrix(1, 1)
		y.Set(0, 0, float64(3.0)*in*in+2+rand.NormFloat64()*sd)

		samples = append(samples, matrix.NewSample(x, y))
	}
	return samples
}

func makeNetwork(layers, hidden, inputs, outputs int, eta, lambda float64, act string) (*matrix.MLann, error) {
	ann := matrix.NewMLann(eta, lambda)
	input := matrix.NewLayer(inputs, hidden, &matrix.Linear{}, &matrix.LinearPrime{}, true)
	err := ann.AddLayer(input)
	if err != nil {
		return nil, err
	}
	for idx := 0; idx < layers; idx++ {
		var activation matrix.Activation
		var derivative matrix.Activation
		if act == "relu" {
			activation = &matrix.RELU{}
			derivative = &matrix.RELUPrime{}
		} else {
			activation = &matrix.Sigmoid{}
			derivative = &matrix.SigmoidPrime{}
		}

		layer := matrix.NewLayer(hidden, hidden, activation, derivative, false)
		err = ann.AddLayer(layer)
		if err != nil {
			return nil, err
		}
	}

	output := matrix.NewLayer(hidden, outputs, &matrix.Linear{}, &matrix.LinearPrime{}, false)
	err = ann.AddLayer(output)
	if err != nil {
		return nil, err
	}
	return ann, nil
}

func makeMatrix(r, c int, data []float64) *matrix.Matrix {
	out := matrix.NewMatrix(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(i, j, data[i*c+j])
		}
	}
	return out
}

func convSamples(in []matrix.Sample, ann *matrix.MLann) [][]float64 {
	data := make([][]float64, 2)
	data[0] = make([]float64, len(in))
	data[1] = make([]float64, len(in))
	for idx := range in {
		data[0][idx], _ = in[idx].Y().Get(0, 0)
		out, _ := ann.FeedForward(in[idx].X())
		data[1][idx], _ = out.Get(0, 0)
	}
	return data
}

func main() {
	//generate the samples
	sd := float64(1.0) / 10.0
	samples := generateData(100, sd)

	//Initialize the network
	ann, err := makeNetwork(2, 4, 1, 1, 1.0, 0.1, "sigmoid")
	if err != nil {
		return
	}

	zero := makeMatrix(1, 1, []float64{0.0})
	middle := makeMatrix(1, 1, []float64{0.5})
	end := makeMatrix(1, 1, []float64{1.0})

	//Train the network
	out0, _ := ann.FeedForward(zero)
	out1, _ := ann.FeedForward(middle)
	out2, _ := ann.FeedForward(end)

	for idx := 0; idx < 1; idx++ {
		fmt.Println("2.0, 2.75, 5.0")
		a, _ := out0.Get(0, 0)
		b, _ := out1.Get(0, 0)
		c, _ := out2.Get(0, 0)
		fmt.Printf("%.8f, %.8f, %.8f\n", a, b, c)

		err = ann.Train(samples, false, 1000)
		if err != nil {
			return
		}
		dist, err2 := ann.SquaredError(samples)
		if err2 != nil {
			return
		}
		fmt.Println(idx, dist)
		out0, _ = ann.FeedForward(zero)
		out1, _ = ann.FeedForward(middle)
		out2, _ = ann.FeedForward(end)

		fmt.Println(ann.String())
	}

	fmt.Println("2.0, 2.75, 5.0")
	a, _ := out0.Get(0, 0)
	b, _ := out1.Get(0, 0)
	c, _ := out2.Get(0, 0)
	fmt.Printf("%.8f, %.8f, %.8f\n", a, b, c)

	if err := ui.Init(); err != nil {
		log.Fatalf("failed to initialize termui: %v", err)
	}
	defer ui.Close()

	plot := widgets.NewPlot()
	plot.Data = convSamples(samples, ann)
	plot.SetRect(0, 0, 120, 50)
	plot.PlotType = widgets.ScatterPlot
	ui.Render(plot)

	for e := range ui.PollEvents() {
		if e.Type == ui.KeyboardEvent {
			break
		}
	}
}
