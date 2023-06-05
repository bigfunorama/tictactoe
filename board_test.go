package tictactoe

import (
	"fmt"
	"testing"

	"bigfunbrewing.com/tensor"
)

func TestDisplay(t *testing.T) {
	b := NewBoard()
	b.Reset()
	b.Display()
}

func TestReset(t *testing.T) {
	b := BoardImp{}
	b.Reset()
	b.Display()
	b.Move(&Move{1, 0, 0})
}

func TestValidateDoubleMove(t *testing.T) {
	b := BoardImp{}
	b.Reset()
	err := b.Move(&Move{1, 0, 0})
	if err != nil {
		t.Errorf(err.Error())
	}
	err = b.Move(&Move{1, 0, 0})
	if err == nil {
		t.Errorf(err.Error())
	}
}

func TestValidateBadMove(t *testing.T) {
	b := BoardImp{}
	b.Reset()
	err := b.Move(&Move{0, 0, 0})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 0, 3})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 0, -1})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, -1, 0})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 3, 0})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 3, 3})
	if err == nil {
		t.Errorf(err.Error())
	}
}

func TestBoardToTensor(t *testing.T) {
	type test struct {
		b *BoardImp
		o *tensor.Tensor[float64]
	}

	tests := []test{
		{
			b: &BoardImp{data: [][]int{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
			o: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking[float64](tensor.Repeat(9, 0.0))),
		},
		{
			b: &BoardImp{data: [][]int{
				{0, 1.0, 0},
				{2.0, 0, 0},
				{0, 0, 0}}},
			o: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking[float64]([]float64{0, 2.0, 0, 1.0, 0, 0, 0, 0, 0})),
		},
	}
	for i := range tests {
		out := tests[i].b.ToPosition()
		if !tests[i].o.Equals(out) {
			t.Errorf("not equal, expected %v, got %v", tests[i].o, out)
		}
		fmt.Println(out)
	}
}
