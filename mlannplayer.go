package tictactoe

import (
	"fmt"

	"bigfunbrewing.com/mlann"
)

// MlannPlayer Uses an existing network to determine moves.
type MlannPlayer struct {
	pid int
	net *mlann.Network
}

func NewMlannPlayer(pid int, net *mlann.Network) *MlannPlayer {
	return &MlannPlayer{net: net, pid: pid}
}

func (mp *MlannPlayer) Move(b Board) (mv *Move, err error) {
	moves, err := ValidMoves(b, mp.pid)
	if err != nil {
		return nil, err
	}
	b.Display()
	for idx := range moves {
		fmt.Println("P:", moves[idx].Pid, "to (", moves[idx].Row, ",", moves[idx].Col, ")")
	}
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
	fmt.Println("picked move (", mv.Row, ",", mv.Col, ")")
	return
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
	fmt.Println("out", out.Shape(), "mv.Row", mv.Row, "mv.Col", mv.Col, "pos", loc(mv.Row, mv.Col))
	if mv.Pid == 1 {
		out.Set(loc(mv.Row, mv.Col), 0, float64(-1))
	}
	if mv.Pid == 2 {
		out.Set(loc(mv.Row, mv.Col), 0, float64(1))
	}

	return out
}

func Update(sample *mlann.Sample) {

}
