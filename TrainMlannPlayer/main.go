package main

import (
	"flag"
	"fmt"
	"os"

	"bigfunbrewing.com/mlann"
	"bigfunbrewing.com/tictactoe"
)

var net1path string
var net2path string
var splayer1 string
var splayer2 string
var episodes int
var gamma float64
var epsilon float64

func init() {
	flag.StringVar(&net1path, "net1", "", "path to the serialized player 1 NN. leave it blank to create a new one")
	flag.StringVar(&net2path, "net2", "", "path to the serialized player 2 NN. leave it blank to create a new one")
	flag.StringVar(&splayer1, "player1", "", "type of player to use for player 1. One of {randoplayer, mlannplayer}")
	flag.StringVar(&splayer2, "player2", "", "type of player to use for player 2. One of {randoplayer, mlannplayer}")
	flag.IntVar(&episodes, "episodes", 10000, "number of games to play with this pair of players")
	flag.Float64Var(&gamma, "gamma", 0.9, "gamma is the discount rate on future rewards")
	flag.Float64Var(&epsilon, "epsilon", 0.01, "epsilon is the exploration rate for NN players")
}

func main() {
	flag.Parse()
	if splayer1 == "" || splayer2 == "" {
		flag.PrintDefaults()
		return
	}
	var player1 tictactoe.Player
	var player2 tictactoe.Player
	var net1 *mlann.Network
	var net2 *mlann.Network
	var mp1 *tictactoe.MlannPlayer
	var mp2 *tictactoe.MlannPlayer

	switch splayer1 {
	case "randoplayer":
		player1 = tictactoe.NewRandomPlayer(1)
	case "mlannplayer":
		if net1path == "" {
			net1path = "player1.net"
		}
		var err error
		net1, err = mlann.LoadNetworkFromFile(net1path)
		if err != nil {
			fmt.Println("failed to load network for player1. Making a new one.", err.Error())
			net1 = mlann.NewNetwork(
				18,             //inputs 18, 9 for the board, 9 for the move
				0.005,          // alpha the learning_rate
				0.3,            // lambda the regularization term for weight decay
				"squarederror", //cost function to use
				1,              // mini batch size (not using cause the samples are so few)
				0,              //thread count for parallelization
				1)              //block size for parallelization
			net1.AddLayer(72, "Dense", "RELU")
			net1.AddLayer(72, "Dense", "RELU")
			net1.AddLayer(36, "Dense", "RELU")
			net1.AddLayer(18, "Dense", "RELU")
			net1.AddLayer(1, "Dense", "Linear")
		} else {
			fmt.Println("found network for player 1.")
		}
		mp1 = tictactoe.NewMlannPlayer(1, epsilon, net1)
		player1 = mp1
	}

	switch splayer2 {
	case "randoplayer":
		player2 = tictactoe.NewRandomPlayer(2)
	case "mlannplayer":
		if net2path == "" {
			net2path = "player2.net"
		}
		var err error
		net2, err = mlann.LoadNetworkFromFile(net2path)
		if err != nil {
			fmt.Println("failed to load network. making a new one.", err.Error())
			net2 = mlann.NewNetwork(
				18,             //inputs 18, 9 for the board, 9 for the move
				0.005,          // alpha the learning_rate
				0.3,            // lambda the regularization term for weight decay
				"squarederror", //cost function to use
				1,              // mini batch size (not using cause the samples are so few)
				0,              //thread count for parallelization
				1)              //block size for parallelization
			net2.AddLayer(72, "Dense", "RELU")
			net2.AddLayer(72, "Dense", "RELU")
			net2.AddLayer(36, "Dense", "RELU")
			net2.AddLayer(18, "Dense", "RELU")
			net2.AddLayer(1, "Dense", "Linear")
		} else {
			fmt.Println("found network for player 2")
		}
		mp2 = tictactoe.NewMlannPlayer(2, epsilon, net2)
		player2 = mp2
	}

	fmt.Println(splayer1, "vs", splayer2)
	trainplayers(player1, player2, episodes, gamma)

	if splayer1 == "mlannplayer" {
		fmt.Println("saving network to file", net1path)
		err := os.Remove(net1path)
		if err != nil {
			fmt.Println(err.Error())
		}
		err = mlann.SaveNetworkToFile(mp1.GetNetwork(), net1path)
		if err != nil {
			fmt.Println("error saving network for player1,", err.Error())
		}
	}

	if splayer2 == "mlannplayer" {
		fmt.Println("saving network to file", net2path)
		err := os.Remove(net2path)
		if err != nil {
			fmt.Println(err.Error())
		}
		err = mlann.SaveNetworkToFile(mp2.GetNetwork(), net2path)
		if err != nil {
			fmt.Println("error saving network for player2.", err.Error())
		}
	}
}

func trainplayers(player1, player2 tictactoe.Player, episodes int, gamma float64) {
	p1rewards := []float64{1.5, -1.0, 1.4}
	p2rewards := []float64{1.5, -1.0, 1.4}

	cone := 0
	ctwo := 0
	cdraw := 0
	var sample1 *mlann.Sample
	var sample2 *mlann.Sample
	bsize := 72
	var sone float64
	var stwo float64
	var sdraw float64
	batches := 0
	for i := 0; i < episodes; i++ {
		//play a game and get the sequence of [board,mv] and who won
		mvs1, mvs2, outcome := episode(player1, player2)
		sample1 = appendSample(sample1, makeSamples(gamma, mvs1, outcome, 1, p1rewards))
		sample2 = appendSample(sample2, makeSamples(gamma, mvs2, outcome, 2, p2rewards))
		switch outcome {
		case 1:
			cone++
		case 2:
			ctwo++
		default:
			cdraw++
		}
		//let's accumulate samples off policy and then train on the collection versus an individual game
		//this will help Adam optimize
		if i > 0 && i%bsize == 0 {
			player1.Train(sample1)
			player2.Train(sample2)
			pone := float64(cone) / float64(bsize)
			ptwo := float64(ctwo) / float64(bsize)
			pdraw := float64(cdraw) / float64(bsize)
			fmt.Printf("%d, %.2f, %.2f, %.2f\n", i, pone, ptwo, pdraw)
			sone += pone
			stwo += ptwo
			sdraw += pdraw
			batches += 1
			cone = 0
			ctwo = 0
			cdraw = 0
			sample1 = nil
			sample2 = nil
		}
	}
	fmt.Printf("final: %d, one: %.2f, two: %.2f, draw: %.2f\n", batches, sone/float64(batches), stwo/float64(batches), sdraw/float64(batches))
}

func appendSample(samples, nsample *mlann.Sample) *mlann.Sample {
	var X *mlann.Matrix
	var Y *mlann.Matrix
	if samples != nil {
		X = samples.X()
		Y = samples.Y()
		X.AppendColumns(nsample.X())
		Y.AppendColumns(nsample.Y())
	} else {
		X = nsample.X()
		Y = nsample.Y()
	}
	return mlann.NewSample(X, Y)
}

func episode(player1, player2 tictactoe.Player) (p1mvs, p2mvs []*mlann.Matrix, outcome int) {
	b := tictactoe.NewBoard()
	b.Reset()
	w := 0
	p1mvs = make([]*mlann.Matrix, 0)
	p2mvs = make([]*mlann.Matrix, 0)
	for w == 0 {
		mv, err := player1.Move(b)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		in := tictactoe.MakeInput(b, mv)
		p1mvs = append(p1mvs, in)

		err = b.Move(mv)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		w = b.GameOver()
		if w == -1 || w == 1 {
			break
		}
		mv, err = player2.Move(b)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		in = tictactoe.MakeInput(b, mv)
		p2mvs = append(p2mvs, in)

		err = b.Move(mv)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		w = b.GameOver()
		if w == -1 || w == 2 {
			break
		}
	}
	outcome = w
	return
}

// rewards is a 3 element slice 0: win, 1: loss, 2: draw
// for each game rotate the samples 90, 180, 270 degrees
// as they are all identical.
func makeSamples(gamma float64, mvs []*mlann.Matrix, outcome, pid int, rewards []float64) *mlann.Sample {
	//compute the reward from the outcome
	reward := float64(0.0)
	if outcome > 0 && outcome == pid {
		//player 1 won
		reward = rewards[0]
	}
	if outcome > 0 && outcome != pid {
		reward = rewards[1]
	}
	if outcome == -1 {
		reward = rewards[2]
	}
	X := mvs[len(mvs)-1]
	Y := mlann.NewMatrix(1, 1)
	y := reward
	Y.Set(0, 0, y)

	for idx := len(mvs) - 2; idx >= 0; idx-- {
		y = gamma*y - 0.10
		X = X.AppendColumns(mvs[idx])
		Y = Y.AppendScalarColumn(y)

		X = X.AppendColumns(rotate(mvs[idx], 1))
		Y = Y.AppendScalarColumn(y)

		X = X.AppendColumns(rotate(mvs[idx], 2))
		Y = Y.AppendScalarColumn(y)

		X = X.AppendColumns(rotate(mvs[idx], 3))
		Y = Y.AppendScalarColumn(y)
	}
	//apply the discounted reward to each state action pair observed in the game and
	//update the network based on the results
	return mlann.NewSample(X, Y)
}

// rotate the board and position by inc*90 degrees. Anything bigger than
// 3 will be interpreted as modulo 4, where any value of 0 returns the
// original.
func rotate(m *mlann.Matrix, inc int) (out *mlann.Matrix) {
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

func evalplayers(mlannplayer, randoplayer tictactoe.Player, episodes int) (win, lose, draw int) {
	i := 0
	for i < episodes {
		//play a game and get the sequence of [board,mv] and who won
		_, _, outcome := episode(mlannplayer, randoplayer)
		if i > 0 && i%200 == 0 {
			fmt.Println(i, "win", win, "lose", lose, "draw", draw)
		}
		switch outcome {
		case -1:
			draw++
		case 1:
			win++
		case 2:
			lose++
		}
		i++
	}
	fmt.Println("win", win, "lose", lose, "draw", draw)
	return
}
