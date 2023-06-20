package tictactoe

import (
	"testing"

	"bigfunbrewing.com/tensor"
)

func TestGruMakeSequenceSample(t *testing.T) {
	type test struct {
		g       *GamePlayed
		pid     int
		rewards []float64
		out     float64
	}
	tests := []test{
		{
			g: &GamePlayed{
				positions: []Position{
					tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
						0, 0, 0,
						0, 0, 0,
						0, 0, 0,
					})),
					tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
						0, 0, 0,
						0, 1, 0,
						0, 0, 0,
					})),
					tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
						2, 0, 0,
						0, 1, 0,
						0, 0, 0,
					})),
					tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
						2, 1, 0,
						0, 1, 0,
						0, 0, 0,
					})),
					tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
						2, 1, 0,
						2, 1, 0,
						0, 0, 0,
					})),
					tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
						2, 1, 0,
						2, 1, 0,
						0, 1, 0,
					})),
				},
				outcome: 1.0,
			},
			pid:     1,
			rewards: []float64{1.0, 0.0, 0.5},
			out:     1.0,
		},
	}
	for i := range tests {
		out := makeSequenceSamples([]*GamePlayed{tests[i].g}, tests[i].pid, tests[i].rewards)
		if tests[i].out != out[0].Y().Get(0, 0) {
			t.Errorf("expected %.2f, got %.2f", tests[i].out, out[0].Y().Get(0, 0))
		}
	}
}
