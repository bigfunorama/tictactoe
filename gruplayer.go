package tictactoe

import (
	"fmt"
	"os"

	"bigfunbrewing.com/tensor"
)

type GruPlayer struct {
	gru    *tensor.Gru[float64]
	output tensor.Layer[float64]
	pid    int
}

// NewPlayer from a previously persisted state. If the provided path is empty
// then GruPlayer makes a new player with random initial state. Gru will be used
// to estimate the reward for a given board position. So the input to Gru for a given
// game would be a reward and a sequence of board positions from the first pair of moves
// to the end of the game.
func NewGruPlayer(pid int, path string) *GruPlayer {
	outputs := 8
	alpha := 0.01
	gp := &GruPlayer{
		gru: tensor.NewGru(
			9, //inputs per word
			outputs,
			"adam",
			alpha,
			0.3,
		),
		output: tensor.NewDense[float64](outputs, 1, 1.0, 0.1, tensor.Relu[float64]{}, "adam", alpha, 0.3),
	}

	if path != "" {
		f, err := os.Open(path)
		if err != nil {
			panic(err.Error())
		}
		gp.gru.Read(f)
		gp.output.Read(f)
	}

	return gp
}

func (gp *GruPlayer) Move(b Board) (mv *Move, err error) {
	return nil, nil
}

func (gp *GruPlayer) Train(games []*GamePlayed) {

}

func (gp *GruPlayer) Display(b Board) {

}

func (gp *GruPlayer) Persist(path string) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err.Error())
	}
	gp.gru.Write(f)
}
