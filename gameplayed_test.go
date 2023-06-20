package tictactoe

import (
	"fmt"
	"os"
	"testing"

	"bigfunbrewing.com/tensor"
)

func TestRotate(t *testing.T) {
	type test struct {
		in  *tensor.Tensor[float64]
		out *tensor.Tensor[float64]
		inc int
	}

	tests := []test{
		{
			in: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				0, 3, 6,
				1, 4, 7,
				2, 5, 8,
			})),
			out: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				2, 1, 0,
				5, 4, 3,
				8, 7, 6,
			})),
			inc: 1,
		},
		{
			in: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				0, 3, 6,
				1, 4, 7,
				2, 5, 8,
			})),
			out: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				8, 5, 2,
				7, 4, 1,
				6, 3, 0,
			})),
			inc: 2,
		},
		{
			in: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				0, 3, 6,
				1, 4, 7,
				2, 5, 8,
			})),
			out: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				6, 7, 8,
				3, 4, 5,
				0, 1, 2,
			})),
			inc: 3,
		},
		{
			in: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				0, 3, 6,
				1, 4, 7,
				2, 5, 8,
			})),
			out: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				0, 3, 6,
				1, 4, 7,
				2, 5, 8,
			})),
			inc: 0,
		},
	}
	for i := range tests {
		o := rotate(tests[i].in, tests[i].inc)
		if !(tests[i].out.Equals(o)) {
			t.Errorf("rotation %d failed", i)
			fmt.Println("expected:")
			tests[i].out.Display(os.Stdout)
			fmt.Println("got:")
			o.Display(os.Stdout)
		}
	}
}
