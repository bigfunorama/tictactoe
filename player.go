package tictactoe

import (
	"math/rand"

	"bigfunbrewing.com/mlann"
)

type Player interface {
	Move(b Board) (mv *Move, err error)
	Train(sample *mlann.Sample)
}

type RandomPlayer struct {
	pid int
}

type Move struct {
	Pid int
	Row int
	Col int
}

type GameOver struct{}

func (g *GameOver) Error() string {
	return "No more moves to make. Game Over."
}

func NewRandomPlayer(pid int) *RandomPlayer {
	return &RandomPlayer{pid: pid}
}

func (rp *RandomPlayer) Move(b Board) (mv *Move, err error) {
	moves, err := ValidMoves(b, rp.pid)
	if err != nil {
		return nil, err
	}
	idx := rand.Intn(len(moves))
	mv = moves[idx]
	return
}

func (rp *RandomPlayer) Train(sample *mlann.Sample) {
	//do nothing
}

func ValidMoves(b Board, pid int) ([]*Move, error) {
	moves := make([]*Move, 0)
	for rr := 0; rr < 3; rr++ {
		for cc := 0; cc < 3; cc++ {
			mv := &Move{Pid: pid, Row: rr, Col: cc}
			if b.Validate(mv) {
				moves = append(moves, mv)
			}
		}
	}
	if len(moves) == 0 {
		return nil, &GameOver{}
	}
	return moves, nil
}
