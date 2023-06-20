package tictactoe

import (
	"fmt"
	"os"
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

func TestBoardGameOver(t *testing.T) {
	type test struct {
		b *BoardImp
		o int
	}

	tests := []test{
		{ //0
			b: &BoardImp{data: [][]int{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}},
			o: 0,
		},
		{ //1 1 wins
			b: &BoardImp{data: [][]int{
				{1, 0, 0},
				{0, 1, 0},
				{2, 2, 1}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //2 1 wins
			b: &BoardImp{data: [][]int{
				{2, 0, 1},
				{0, 1, 0},
				{1, 2, 0}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //3 2 wins
			b: &BoardImp{data: [][]int{
				{2, 0, 0},
				{1, 2, 0},
				{1, 1, 2}},
				g: NewGamePlayed()},
			o: 2,
		},
		{ //4 1 wins
			b: &BoardImp{data: [][]int{
				{1, 1, 1},
				{0, 2, 0},
				{0, 2, 0}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //5 1 wins
			b: &BoardImp{data: [][]int{
				{0, 2, 0},
				{1, 1, 1},
				{0, 2, 0}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //6 1 wins
			b: &BoardImp{data: [][]int{
				{0, 2, 0},
				{0, 2, 0},
				{1, 1, 1}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //7 1 wins
			b: &BoardImp{data: [][]int{
				{1, 2, 0},
				{1, 0, 0},
				{1, 2, 0}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //8 1 wins
			b: &BoardImp{data: [][]int{
				{0, 1, 0},
				{2, 1, 2},
				{0, 1, 0}},
				g: NewGamePlayed()},
			o: 1,
		},
		{ //9 tie
			b: &BoardImp{data: [][]int{
				{1, 2, 1},
				{2, 1, 1},
				{2, 1, 2}},
				g: NewGamePlayed()},
			o: -1,
		},
	}
	for i := range tests {
		if tests[i].o != tests[i].b.GameOver() {
			t.Errorf("%d, expected %d, got %d", i, tests[i].o, tests[i].b.GameOver())
		}
	}
}

func TestBoardMakePosition(t *testing.T) {
	type test struct {
		b Board
		m *Move
		o *tensor.Tensor[float64]
	}
	tests := []test{
		{
			b: &BoardImp{data: [][]int{
				{2, 0, 0},
				{1, 2, 2},
				{1, 1, 0}},
				g: NewGamePlayed()},
			m: &Move{Pid: 1, Row: 0, Col: 1},
			o: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				2, 1, 1, 1, 2, 1, 0, 2, 0,
			})),
		},
		{
			b: &BoardImp{data: [][]int{
				{2, 0, 0},
				{1, 2, 0},
				{1, 1, 0}},
				g: NewGamePlayed()},
			m: &Move{Pid: 2, Row: 2, Col: 2},
			o: tensor.New(tensor.WithShape[float64](9, 1), tensor.WithBacking([]float64{
				2, 1, 1, 0, 2, 1, 0, 0, 2,
			})),
		},
	}

	for i := range tests {
		o := MakePosition(tests[i].b, tests[i].m)
		if !tests[i].o.Equals(o) {
			t.Errorf("%d, failed", i)
			fmt.Println("expected")
			tests[i].o.Display(os.Stdout)
			fmt.Println()
			fmt.Println("got")
			(*tensor.Tensor[float64])(o).Display(os.Stdout)
		}
	}
}
