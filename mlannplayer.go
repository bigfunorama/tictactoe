package tictactoe

import (
	"fmt"
	"math/rand"
	"os"

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
	alpha := 0.05
	lambda := 0.3
	net = tensor.NewNetwork(
		tensor.SquaredError[float64],
		tensor.SquaredErrorPrime[float64],
		50,
		tensor.NewDense[float64](18, 36, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		tensor.NewDense[float64](36, 36, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		tensor.NewDense[float64](36, 36, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
		//tensor.NewDense[float64](36, 36, 1.0, 0.1, tensor.LeakyRelu[float64]{Leak: 0.1}, "adam", alpha, lambda),
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

// Move selects the next move for the player based on the current state.
// This implementation corresponds to the TD(0) implementation I believe.
// So from the current state we look ahead to the next move based on the
// data we've collected so far and select the epsilon-greedy next move
// forecasted by our model.
//
// Alternatively, we can conduct a kind of beam search through the
// possible states to narrow our early searches and state action value
// evaluations. Notice that at the beginning of the game player 1 has to
// evaluate 9 positions. For each position, we don't know what player 2
// will select, that would leave 8 possible new states for each of the
// nine start states to be evaluated. To look, just one move ahead would
// require over 51 million states to be evaluated. If we look at the
// game as a tree where at each level of the tree two more spaces have
// been occupied, one by player 1 and one by player 2, then we see that
// the branching factor is reduced by 2 as you descend down the tree to
// the end game. That is, at the start of the game there are 9 positions
// to evaluate, then 7 at the next move, then 5, 3, and finally 1.
//
// I propose that we evaluate all available moves at the current level,
// then for the best n_1 moves, we expand n_2 available moves for the next
// player. We pick the move with the highest expected outcome after the
// next round of moves.
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
	out := mp.net.Forward(X)
	return out.Get(0, 0)
}

func (mp *MlannPlayer) Train(games []*GamePlayed) {
	sample := makeSamples(mp.gamma, games, mp.pid, []float64{10.0, -10.0, 0.1})
	for i := 0; i < 1; i++ {
		mp.net.Iterate(sample)
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
			zero = fmt.Sprintf("%.8f", v)
		}
		o, _ := b.Get(i, 1)
		one := convert(o)
		if one == "" {
			v := mp.evalMove(b, &Move{Pid: mp.pid, Row: i, Col: 1})
			one = fmt.Sprintf("%.8f", v)
		}
		t, _ := b.Get(i, 2)
		two := convert(t)
		if two == "" {
			v := mp.evalMove(b, &Move{Pid: mp.pid, Row: i, Col: 2})
			two = fmt.Sprintf("%.8f", v)
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
//
// To account for Q-learning I think we have to construct a
// sequence of updates and that this has to proceed as
// stochastic gradient descent.
//
// Let's assume that player 1 wins the game in three moves, the shortest
// possible victory. What are the update equations for the three states?
// Given that the first state action pair (s_0, a_0) then the first update
// equation would be as follows,
//
// Q(s_0, a_0) <- Q(s_0, a_0) + alpha*[r_0 + gamma*Q(s_1, a_1) - Q(s_0, a_0)]
//
// Notice that the update depends on the value of state action pair (s_1, a_1)
// whose update equation looks as follows,
//
// Q(s_1, a_1) <- Q(s_1, a_1) + alpha*[r_1 + gamma*Q(s_2, a_2) - Q(s_1, a_1)]
//
// And so on to (s_2, a_2).
// Q(s_2, a_2) <- Q(s_2, a_2) + alpha*[r_2 + gamma*Q(s_3, a_3) - Q(s_2, a_2)]
//
// Clearly, each game played from length 3 to length 5 for player 1 will have
// similar interior update states where the number of updates matches the number
// of moves that player 1 made during the course of the game. Note that by definition,
// state action pair (s_2, a_2) results in a terminal state where the game has
// ended, meaning that Q(s_3, a_3) is 0 by assumption because the game is over
// and reward r_2 corresponds to the terminal reward for winning the game. This
// means that the entire update for Q(s_2, a_2) is now fully defined and we can
// compute the update for Q(s_2, a_2). That is, r_2 is as provided, Q(s_2, a_2)
// is the result of a forward pass through the network for (s_2, a_2) and alpha,
// gamma are provided.
//
// Based on our update equations, the TD error for Q(s_2, a_2) would be
// delta_2 = r_2 - Q(s_2, a_2). In our mlannplayer, Q(s_2, a_2) is approximated
// by a forward pass through our network, yhat.
//
//
// Working backwards, we would compute the update sample for Q(s_1, a_1) using
// gamma and our known reward and so on, back to Q(s_0, a_0). Updating Q(s_1, a_1),
// r_1 is 0 by definition, because we have an outcome for the episode we know
//

// The key for
// SARSA is to apply the samples to the function approximator one at a time
// iteratively improving the agent, as opposed to applying them all at once through
// a Mini batch process. Unless, we're executing in an off-policy approach where
// we accumulate games and then build training samples and execute one update.
func makeSamples(gamma float64, g []*GamePlayed, pid int, rewards []float64) (out *tensor.Sample[float64]) {
	for i := range g {
		// get the positions from the game for our pid
		gp := NewGamePlayed()

		// player 1 goes first so positions 0,2,4,6,8 are theirs
		start := 0
		// player 2 positions are 1,3,5,7
		if pid == 2 {
			start = 1
		}
		for j := start; j < len(g[i].Positions()); j += 2 {
			gp.Append(g[i].Positions()[j])
		}

		//compute the final reward from the outcome for a win, loss, or tie
		reward := float64(0.0)
		if g[i].outcome > 0 && g[i].outcome == float64(pid) {
			// win
			reward = rewards[0] / float64(len(gp.Positions()))
		}
		if g[i].outcome > 0 && g[i].outcome != float64(pid) {
			// loss
			reward = rewards[1]
		}
		if g[i].outcome == -1 {
			// tie
			reward = rewards[2]
		}

		// discount rewards back in time based on our expected update
		rewards := make([]float64, len(gp.Positions()))
		for j := len(gp.Positions()) - 1; j >= 0; j-- {
			reward *= gamma
			rewards[j] = reward
		}
		if out == nil {
			out = gp.ToSample(rewards)
			if false {
				fmt.Println(rewards)
				out.Y().Display(os.Stdout)
				out.X().Display(os.Stdout)
			}
			for i := 1; i < 4; i++ {
				out.Append(gp.Rotate(i).ToSample(rewards))
			}
		} else {
			out.Append(gp.ToSample(rewards))
			for i := 1; i < 4; i++ {
				out.Append(gp.Rotate(i).ToSample(rewards))
			}
		}
	}

	return
}

func TDError[T tensor.Numeric](yhat, y *tensor.Tensor[T]) *tensor.Tensor[T] {
	return y.Sub(yhat)
}
