package tictactoe

import (
	"fmt"
	"math/rand"
	"os"

	"bigfunbrewing.com/tensor"
)

type GruPlayer struct {
	gru     *tensor.Gru[float64]
	output  tensor.Layer[float64]
	pid     int
	epsilon float64
}

// NewPlayer from a previously persisted state. If the provided path is empty
// then GruPlayer makes a new player with random initial state. Gru will be used
// to estimate the reward for a given board position. So the input to Gru for a given
// game would be a reward and a sequence of board positions from the first pair of moves
// to the end of the game.
func NewGruPlayer(pid int, path string, epsilon float64) *GruPlayer {
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
		if err == nil {
			gp.gru.Read(f)
			gp.output.Read(f)
		}
	}

	return gp
}

func (gp *GruPlayer) Move(b Board) (mv *Move, err error) {
	moves, err := ValidMoves(b, gp.pid)
	if err != nil {
		return nil, err
	}
	if rand.Float64() < gp.epsilon {
		mv, err = (&RandomPlayer{pid: gp.pid}).Move(b)
	} else {
		max := 0.0
		pos := 0
		for i := range moves {
			X := MakePosition(b, moves[i])
			sentence := b.GamePlayed().Positions()
			sentence = append(sentence, X)
			gp.gru.NewSentence(len(sentence), (*tensor.Tensor[float64])(sentence[0]).Shape()[1])
			var yhat *tensor.Tensor[float64]
			for j := range sentence {
				yhat = gp.gru.Forward(sentence[j])
			}
			yhat = gp.output.Forward(yhat)
			if yhat.Get(0, 0) > max {
				max = yhat.Get(0, 0)
				pos = i
			}
		}
		mv = moves[pos]
	}
	return
}

func (gp *GruPlayer) Train(games []*GamePlayed) {
	//convert games played into sentences
	ss := makeSequenceSamples(games, gp.pid, []float64{1.5, -1.4, 1.0})

	//Train the network on those sentences
	for i := 0; i < 10; i++ {
		for k := 0; k < len(ss); k++ {
			x := ss[k].X()
			y := ss[k].Y()
			gp.gru.NewSentence(len(x), x[0].Shape()[1])

			// 1. run the games of length len(x) through the network to produce an estimate
			var yhat *tensor.Tensor[float64]
			for j := range x {
				yhat = gp.gru.Forward(x[j])
			}
			yhat = gp.output.Forward(yhat)

			// 2. Measure the Error and send that back through the network to update the gradients
			err := tensor.SquaredErrorPrime(yhat, y)
			err = gp.output.Backward(err)
			for j := len(x) - 1; j >= 0; j-- {
				err = gp.gru.Backward(err)
			}

			// 3. Update the network
			gp.gru.Update()
		}
	}
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

// makeSequenceSamples converts a slice of GamePlayed into a slice of SequenceSample
// this means converting the game outcome into a reward and setting Y in the SequenceSample
// and the X in SequenceSample to the GamePlayed position sequence. Finally, because each
// game may be a different length the SequenceSamples are sorted based on game length and
// merged leaving the set of games of varying lengths to be returned.
func makeSequenceSamples(g []*GamePlayed, pid int, rewards []float64) (out []*tensor.SequenceSample) {
	tmp := make(map[int]*tensor.SequenceSample)
	for i := range g {
		reward := float64(0.0)
		if g[i].outcome > 0 && g[i].outcome == float64(pid) {
			//player 1 won
			reward = rewards[0]
		}
		if g[i].outcome > 0 && g[i].outcome != float64(pid) {
			reward = rewards[1]
		}
		if g[i].outcome == -1 {
			reward = rewards[2]
		}
		ss := g[i].ToSequenceSample(reward)
		ssl := len(ss.X())
		if _, ok := tmp[ssl]; ok {
			tmp[ssl].Merge(g[i].ToSequenceSample(reward))
		} else {
			tmp[ssl] = g[i].ToSequenceSample(reward)
		}
	}
	out = make([]*tensor.SequenceSample, 0)
	for _, v := range tmp {
		out = append(out, v)
	}
	return
}
