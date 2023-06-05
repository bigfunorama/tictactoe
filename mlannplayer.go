package tictactoe

import (
	"fmt"
	"math/rand"
	"os"

	"bigfunbrewing.com/mlann"
	"bigfunbrewing.com/tensor"
)

// MlannPlayer Uses an existing network to determine moves.
type MlannPlayer struct {
	// pid is the id of the player either 1 or 2
	pid     int
	epsilon float64
	gamma   float64
	net     *tensor.Network[float64]
}

func NewMlannPlayer(pid int, path string, epsilon, gamma float64) *MlannPlayer {
	var net *tensor.Network[float64]
	alpha := 0.1
	lambda := 0.3
	net = tensor.NewNetwork(
		tensor.SquaredError[float64],
		tensor.SquaredErrorPrime[float64],
		100,
		tensor.NewDense[float64](18, 72, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		tensor.NewDense[float64](72, 72, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		tensor.NewDense[float64](72, 36, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		tensor.NewDense[float64](36, 18, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		tensor.NewDense[float64](18, 1, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
	)

	if path != "" {
		f, err := os.Open(path)
		if err == nil {
			net.Read(f)
		} else {
			fmt.Println(err.Error())
		}
	}
	return &MlannPlayer{pid: pid, epsilon: epsilon, gamma: gamma, net: net}
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
		X := MakePosition(b, moves[0])
		out := mp.net.Forward(X)
		v := out.Get(0, 0)
		mv = moves[0]
		for idx := 1; idx < len(moves); idx++ {
			X := MakePosition(b, moves[idx])
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
	X := MakePosition(b, mv)
	fmt.Println(X)
	out := mp.net.Forward(X)
	return out.Get(0, 0)
}

func (mp *MlannPlayer) Train(games []*GamePlayed) {
	sample := makeSamples(mp.gamma, games, mp.pid, []float64{1.5, -1.4, 1.0})
	for i := 0; i < 10; i++ {
		mp.net.MiniBatch(sample)
		//yhat := mp.net.Forward(sample.X())
		//diff := yhat.Sub(sample.Y())
		//err := diff.Hadamard(diff).Sum()
		//fmt.Println(i, math.Sqrt(err.Get(0, 0)))
	}
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

func (mp *MlannPlayer) Persist(path string) {
	fmt.Println("saving network to file", path)
	err := os.Remove(path)
	if err != nil {
		fmt.Println(err.Error())
	}
	f, err := os.Create(path)
	if err != nil {
		fmt.Println("error saving network for player1,", err.Error())
	}
	mp.net.Write(f)
}

// rewards is a 3 element slice 0: win, 1: loss, 2: draw
// for each game rotate the samples 90, 180, 270 degrees
// as they are all identical.
func makeSamples(gamma float64, g []*GamePlayed, pid int, rewards []float64) (out *tensor.Sample[float64]) {
	for i := range g {
		// get the positions from the game for our pid
		gp := NewGamePlayed()
		start := 0
		if pid == 2 {
			start = 1
		}
		// player 1 goes first so capture all of the odd positions
		for j := start; j < len(g[i].Positions()); j += 2 {
			gp.Append(g[i].Positions()[j])
		}

		//compute the reward from the outcome
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
		rewards := make([]float64, len(gp.Positions()))
		for j := len(gp.Positions()) - 1; j >= 0; j-- {
			reward *= gamma
			rewards[j] = reward
		}
		if out == nil {
			out = gp.ToSample(rewards)
		} else {
			out.Append(gp.ToSample(rewards))
		}
	}

	return
}

// rotate the board and position by inc*90 degrees. Anything bigger than
// 3 will be interpreted as modulo 4, where any value of 0 returns the
// original.
func rotateOld(m *mlann.Matrix, inc int) (out *mlann.Matrix) {
	out = mlann.NewMatrix(m.Rows(), m.Cols())
	switch inc % 4 {
	case 1: //90 degrees
		//0,1,2     6,3,0
		//3,4,5 ->  7,4,1
		//6,7,8     8,5,2
		out = mlann.NewMatrix(18, 1)
		out.Set(0, 0, m.Get(6, 0))
		out.Set(1, 0, m.Get(3, 0))
		out.Set(2, 0, m.Get(0, 0))
		out.Set(3, 0, m.Get(7, 0))
		out.Set(4, 0, m.Get(4, 0))
		out.Set(5, 0, m.Get(1, 0))
		out.Set(6, 0, m.Get(8, 0))
		out.Set(7, 0, m.Get(5, 0))
		out.Set(8, 0, m.Get(2, 0))

		out.Set(9, 0, m.Get(6+9, 0))
		out.Set(10, 0, m.Get(3+9, 0))
		out.Set(11, 0, m.Get(0+9, 0))
		out.Set(12, 0, m.Get(7+9, 0))
		out.Set(13, 0, m.Get(4+9, 0))
		out.Set(14, 0, m.Get(1+9, 0))
		out.Set(15, 0, m.Get(8+9, 0))
		out.Set(16, 0, m.Get(5+9, 0))
		out.Set(17, 0, m.Get(2+9, 0))
	case 2: //180 degrees
		//0,1,2    8,7,6
		//3,4,5 -> 5,4,3
		//6,7,8    2,1,0
		out.Set(0, 0, m.Get(8, 0))
		out.Set(1, 0, m.Get(7, 0))
		out.Set(2, 0, m.Get(6, 0))
		out.Set(3, 0, m.Get(5, 0))
		out.Set(4, 0, m.Get(4, 0))
		out.Set(5, 0, m.Get(3, 0))
		out.Set(6, 0, m.Get(2, 0))
		out.Set(7, 0, m.Get(1, 0))
		out.Set(8, 0, m.Get(0, 0))

		out.Set(0, 0, m.Get(8+9, 0))
		out.Set(1, 0, m.Get(7+9, 0))
		out.Set(2, 0, m.Get(6+9, 0))
		out.Set(3, 0, m.Get(5+9, 0))
		out.Set(4, 0, m.Get(4+9, 0))
		out.Set(5, 0, m.Get(3+9, 0))
		out.Set(6, 0, m.Get(2+9, 0))
		out.Set(7, 0, m.Get(1+9, 0))
		out.Set(8, 0, m.Get(0+9, 0))
	case 3: //270 degrees
		//0,1,2    2,5,8
		//3,4,5 -> 1,4,7
		//6,7,8    0,3,6
		out.Set(0, 0, m.Get(2, 0))
		out.Set(1, 0, m.Get(5, 0))
		out.Set(2, 0, m.Get(8, 0))
		out.Set(3, 0, m.Get(1, 0))
		out.Set(4, 0, m.Get(4, 0))
		out.Set(5, 0, m.Get(7, 0))
		out.Set(6, 0, m.Get(0, 0))
		out.Set(7, 0, m.Get(3, 0))
		out.Set(8, 0, m.Get(6, 0))

		out.Set(0, 0, m.Get(2+9, 0))
		out.Set(1, 0, m.Get(5+9, 0))
		out.Set(2, 0, m.Get(8+9, 0))
		out.Set(3, 0, m.Get(1+9, 0))
		out.Set(4, 0, m.Get(4+9, 0))
		out.Set(5, 0, m.Get(7+9, 0))
		out.Set(6, 0, m.Get(0+9, 0))
		out.Set(7, 0, m.Get(3+9, 0))
		out.Set(8, 0, m.Get(6+9, 0))
	default:
		out = m.Clone()
	}
	return
}
