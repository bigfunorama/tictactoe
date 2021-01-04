package model

import "math/rand"

type Player interface {
	Move(b Board) (mv *Move, err error)
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
	moves := make([]*Move, 0)
	for rr := 0; rr < 3; rr++ {
		for cc := 0; cc < 3; cc++ {
			mv := &Move{Pid: rp.pid, Row: rr, Col: cc}
			if b.Validate(mv) {
				moves = append(moves, mv)
			}
		}
	}
	if len(moves) == 0 {
		return nil, &GameOver{}
	}
	idx := rand.Intn(len(moves))
	mv = moves[idx]
	return
}
