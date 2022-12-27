package tictactoe

import (
	"fmt"
	"math/rand"

	"bigfunbrewing.com/mlann"
)

// MlannPlayer Uses an existing network to determine moves.
type MlannPlayer struct {
	pid     int
	epsilon float64
	net     *mlann.Network
}

func NewMlannPlayer(pid int, epsilon float64, net *mlann.Network) *MlannPlayer {
	return &MlannPlayer{pid: pid, epsilon: epsilon, net: net}
}

func (mp *MlannPlayer) SetEpsilon(epsilon float64) {
	mp.epsilon = epsilon
}

func (mp *MlannPlayer) Move(b Board) (mv *Move, err error) {
	moves, err := ValidMoves(b, mp.pid)
	if err != nil {
		return nil, err
	}
	if rand.Float64() < mp.epsilon {
		mv, err = (&RandomPlayer{pid: mp.pid}).Move(b)
	} else {
		X := MakeInput(b, moves[0])
		out := mp.net.Forward(X)
		v := out.Get(0, 0)
		mv = moves[0]
		for idx := 1; idx < len(moves); idx++ {
			X := MakeInput(b, moves[idx])
			out := mp.net.Forward(X)
			if out.Get(0, 0) > v {
				v = out.Get(0, 0)
				mv = moves[idx]
			}
		}
	}
	return
}

func (mp *MlannPlayer) evalMove(b Board, mv *Move) float64 {
	X := MakeInput(b, mv)
	out := mp.net.Forward(X)
	return out.Get(0, 0)
}

func (mp *MlannPlayer) Train(sample *mlann.Sample) {
	mp.net = mp.net.Adam(sample)
}

func convert(player int) (out string) {
	if player == 1 {
		out = "    X    "
	}
	if player == 2 {
		out = "    O    "
	}
	return
}
func (mp *MlannPlayer) Display(b Board) {
	fmt.Println("         |     0     |     1     |     2     ")
	fmt.Println("---------+-----------+-----------+-----------")
	for i := 0; i < 3; i++ {
		z, _ := b.Get(i, 0)
		zero := convert(z)
		if zero == "" {
			v := mp.evalMove(b, &Move{Pid: mp.pid, Row: i, Col: 0})
			zero = fmt.Sprintf("%.5f", v)
		}
		o, _ := b.Get(i, 1)
		one := convert(o)
		if one == "" {
			v := mp.evalMove(b, &Move{Pid: mp.pid, Row: i, Col: 1})
			one = fmt.Sprintf("%.5f", v)
		}
		t, _ := b.Get(i, 2)
		two := convert(t)
		if two == "" {
			v := mp.evalMove(b, &Move{Pid: mp.pid, Row: i, Col: 2})
			two = fmt.Sprintf("%.5f", v)
		}
		fmt.Printf("    %d    | %s | %s | %s\n", i, zero, one, two)
		if i < 2 {
			fmt.Println("---------+---------+---------+---------")
		} else {
			fmt.Println()
		}
	}
}

func MakeInput(b Board, mv *Move) *mlann.Matrix {
	out := mlann.NewMatrix(9, 1)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			p, _ := b.Get(i, j)
			if p == 1 {
				out.Set(loc(i, j), 0, float64(-1))
			}
			if p == 2 {
				out.Set(loc(i, j), 0, float64(1))
			}
		}
	}
	pos := convertMove(mv)
	out = out.AppendRows(pos)
	return out
}

func loc(r, c int) int {
	return r*3 + c
}

func convertMove(mv *Move) *mlann.Matrix {
	out := mlann.NewMatrix(9, 1)
	if mv.Pid == 1 {
		out.Set(loc(mv.Row, mv.Col), 0, float64(-1))
	}
	if mv.Pid == 2 {
		out.Set(loc(mv.Row, mv.Col), 0, float64(1))
	}

	return out
}
