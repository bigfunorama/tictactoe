package tictactoe

import (
	"fmt"
	"math/rand"
	"os"

	"bigfunbrewing.com/tensor"
)

type GruPlayer struct {
	gru     *tensor.Gru[float64]
	output  *tensor.Network[float64]
	pid     int
	epsilon float64
}

// NewPlayer from a previously persisted state. If the provided path is empty
// then GruPlayer makes a new player with random initial state. Gru will be used
// to estimate the reward for a given board position. So the input to Gru for a given
// game would be a reward and a sequence of board positions from the first pair of moves
// to the end of the game. Gru would play each game from the beginning, generating
// new board states  from the positions already played. But, how would we convert the
// continuous output of the gru as board positions? Not sure how this would work yet.
func NewGruPlayer(pid int, path string, epsilon float64) *GruPlayer {
	outputs := 36
	alpha := 0.01
	lambda := 0.3
	gp := &GruPlayer{
		pid:     pid,
		epsilon: epsilon,
		gru: tensor.NewGru(
			18, //inputs per word 9 for just the board, 18 for board plus move
			outputs,
			"adam",
			alpha,
			0.3,
		),
		output: tensor.NewNetwork(
			tensor.SquaredError[float64],
			tensor.SquaredErrorPrime[float64],
			100,
			tensor.NewDense[float64](36, 18, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", 0.1, lambda),
			tensor.NewDense[float64](18, 1, 1.0, 0.1, tensor.Linear[float64]{}, "adam", 0.1, lambda),
		),
	}

	if path != "" {
		if _, err := os.Stat(path); err == nil {
			f, err := os.Open(path)
			if err == nil {
				gp.gru.Read(f)
				gp.output.Read(f)
			}
		}
	}

	return gp
}

// This model kind of sucks cause it's picking a next move
// and not a sequence of moves based on the current state.
// But Gru was trained on complete games, where we treated
// each state as a word in the sentence of a game. Instead of
// producing just a single outcome from a given state, we should
// use GRU to play out the game from the current state and
// look for the best outcome.
//
// To get to this style of play, we pass the current Position (s_i, a_i)
// through the network and pass the network output as the next
// input to gru until
func (gp *GruPlayer) Move(b Board) (mv *Move, err error) {
	moves, err := ValidMoves(b, gp.pid)
	if err != nil {
		fmt.Println("gru found no valid moves")
		return nil, err
	}
	if rand.Float64() < gp.epsilon {
		//fmt.Printf(".")
		mv, err = (&RandomPlayer{pid: gp.pid}).Move(b)
	} else {
		yhat := gp.evalMove(b, moves[0])
		max := yhat
		pos := 0
		for i := 1; i < len(moves); i++ {
			yhat = gp.evalMove(b, moves[i])
			if yhat > max {
				max = yhat
				pos = i
				//fmt.Println("max move", max, moves[pos])
			}
		}
		mv = moves[pos]
	}
	return
}

func (gp *GruPlayer) Train(games []*GamePlayed) {
	//convert games played into sentences
	ss := makeSequenceSamples(games, gp.pid, []float64{2.0, -2.0, 0.0})

	//Train the network on those sentences
	for i := 0; i < 1; i++ {
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
	fmt.Println("         |     0     |     1     |     2     ")
	fmt.Println("---------+-----------+-----------+-----------")
	for i := 0; i < 3; i++ {
		z, _ := b.Get(i, 0)
		zero := convert(z)
		if zero == "" {
			v := gp.evalMove(b, &Move{Pid: gp.pid, Row: i, Col: 0})
			zero = fmt.Sprintf("%.5f", v)
		}
		o, _ := b.Get(i, 1)
		one := convert(o)
		if one == "" {
			v := gp.evalMove(b, &Move{Pid: gp.pid, Row: i, Col: 1})
			one = fmt.Sprintf("%.5f", v)
		}
		t, _ := b.Get(i, 2)
		two := convert(t)
		if two == "" {
			v := gp.evalMove(b, &Move{Pid: gp.pid, Row: i, Col: 2})
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

func (gp *GruPlayer) evalMove(b Board, mv *Move) float64 {
	X := MakePosition(b, mv)
	sentence := b.GamePlayed().Positions()
	sentence = append(sentence, X)
	gp.gru.NewSentence(len(sentence), (*tensor.Tensor[float64])(sentence[0]).Shape()[1])
	var yhat *tensor.Tensor[float64]
	for j := range sentence {
		yhat = gp.gru.Forward(sentence[j])
	}
	yhat = gp.output.Forward(yhat)
	return yhat.Get(0, 0)
}

func (gp *GruPlayer) Persist(path string) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err.Error())
	}
	gp.gru.Write(f)
	gp.output.Write(f)
}

// makeSequenceSamples converts a slice of GamePlayed into a slice of SequenceSample
// this means converting the game outcome into a reward and setting Y in the SequenceSample
// and the X in SequenceSample to the GamePlayed position sequence. Finally, because each
// game may be a different length the SequenceSamples are sorted based on game length and
// merged leaving the set of games of varying lengths to be returned.
func makeSequenceSamples(g []*GamePlayed, pid int, rewards []float64) (out []*tensor.SequenceSample) {
	tmp := make(map[int]*tensor.SequenceSample)
	for i := range g {
		gp := NewGamePlayed()
		start := 1
		if pid == 2 {
			start = 2
		}
		// player 1 goes first so capture all of the odd positions
		for j := start; j < len(g[i].Positions()); j += 2 {
			gp.Append(g[i].Positions()[j])
		}

		reward := float64(0.0)
		if g[i].Outcome() > 0 && g[i].Outcome() == float64(pid) {
			reward = rewards[0] * 3.0 / float64(len(gp.Positions()))
		}
		if g[i].Outcome() > 0 && g[i].Outcome() != float64(pid) {
			reward = rewards[1]
		}
		if g[i].Outcome() == -1 {
			reward = rewards[2]
		}
		for i := 0; i < 4; i++ {
			ss := gp.Rotate(i).ToSequenceSample(reward)
			ssl := len(ss.X())
			if _, ok := tmp[ssl]; ok {
				tmp[ssl].Merge(ss)
			} else {
				tmp[ssl] = ss
			}
		}
	}
	out = make([]*tensor.SequenceSample, 0)
	for _, v := range tmp {
		out = append(out, v)
	}
	return
}
