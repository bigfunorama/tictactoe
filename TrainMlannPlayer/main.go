package main

import (
	"fmt"
	"os"

	"bigfunbrewing.com/mlann"
	"bigfunbrewing.com/tictactoe"
)

func main() {
	// make a network
	net := mlann.NewNetwork(
		18,             //inputs 18, 9 for the board, 9 for the move
		0.05,           // alpha the learning_rate
		0.3,            // lambda the regularization term for weight decay
		"squarederror", //cost function to use
		1,              // mini batch size (not using cause the samples are so few)
		32,             //thread count for parallelization
		18)             //block size for parallelization
	net.AddLayer(36, "Dense", "RELU")
	net.AddLayer(72, "Dense", "RELU")
	net.AddLayer(72, "Dense", "RELU")
	net.AddLayer(36, "Dense", "RELU")
	net.AddLayer(1, "Dense", "Linear")

	net2 := mlann.NewNetwork(
		18,             //inputs 18, 9 for the board, 9 for the move
		0.05,           // alpha the learning_rate
		0.3,            // lambda the regularization term for weight decay
		"squarederror", //cost function to use
		1,              // mini batch size (not using cause the samples are so few)
		32,             //thread count for parallelization
		18)             //block size for parallelization
	net2.AddLayer(36, "Dense", "RELU")
	net2.AddLayer(72, "Dense", "RELU")
	net2.AddLayer(72, "Dense", "RELU")
	net2.AddLayer(36, "Dense", "RELU")
	net2.AddLayer(1, "Dense", "Linear")

	//set the decay on future rewards
	gamma := 0.8
	//set the exploration rate
	epsilon := 0.0

	//make the players
	mlannplayer1 := tictactoe.NewMlannPlayer(1, epsilon, net)
	mlannplayer2 := tictactoe.NewMlannPlayer(2, epsilon, net2)
	randoplayer1 := tictactoe.NewRandomPlayer(1)
	randoplayer2 := tictactoe.NewRandomPlayer(2)

	fmt.Println("Initial results")
	win, lose, draw := evalplayers(mlannplayer1, randoplayer2, 1000)
	fmt.Println("\nplayer1 initial win", win, "loss", lose, "draw", draw)
	fmt.Printf("player1 ending win %.2f\n", float64(win)/float64(win+lose+draw))

	win, lose, draw = evalplayers(randoplayer1, mlannplayer2, 1000)
	fmt.Println("\nplayer2 initial win", win, "loss", lose, "draw", draw)
	fmt.Printf("player2 ending win %.2f\n", float64(win)/float64(win+lose+draw))

	//train polayer 1 on a random partner
	epsilon = 0.15
	fmt.Println("train player 1")
	mlannplayer1.SetEpsilon(epsilon)
	trainplayers(mlannplayer1, randoplayer2, 50000, gamma)

	//train player 2 on a random player
	fmt.Println("\ntrain player 2")
	mlannplayer2.SetEpsilon(epsilon)
	trainplayers(randoplayer1, mlannplayer2, 50000, gamma)

	//train player 1 on player 2
	fmt.Println("play1 v player2")
	trainplayers(mlannplayer1, mlannplayer2, 300000, gamma)

	//evaluate player 1 and player 2 on random players
	epsilon = 0.0
	fmt.Println("\neval player1")
	win, lose, draw = evalplayers(mlannplayer1, randoplayer2, 1000)
	fmt.Println("\nplayer1 win", win, "loss", lose, "draw", draw)
	fmt.Printf("player1 ending win %.2f\n", float64(win)/float64(win+lose+draw))

	f, err := os.Create("player1.net")
	if err != nil {
		fmt.Println("player 1 failed to write", err.Error())
	} else {
		defer f.Close()
		net.Write(f)
	}

	win, lose, draw = evalplayers(randoplayer1, mlannplayer2, 1000)
	fmt.Println("\nplayer2 win", win, "loss", lose, "draw", draw)
	fmt.Printf("player2 ending win %.2f\n", float64(win)/float64(win+lose+draw))

	f2, err := os.Create("player2.net")
	if err != nil {
		fmt.Println("player2 failed to write", err.Error())
	} else {
		defer f2.Close()
		net2.Write(f2)
	}
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

func trainplayers(player1, player2 tictactoe.Player, episodes int, gamma float64) {
	i := 0
	one := 0
	two := 0
	draw := 0
	for i < episodes {
		//play a game and get the sequence of [board,mv] and who won
		mvs1, mvs2, outcome := episode(player1, player2)
		switch outcome {
		case 1:
			one++
		case 2:
			two++
		default:
			draw++
		}
		if i > 0 && i%200 == 0 {
			fmt.Println(i, " 1", one, "2", two, "draw", draw)
		}
		//produce updates for player1
		sample1 := makeSamples(gamma, mvs1, outcome, 1)
		player1.Train(sample1)

		//produce updates for player2
		sample2 := makeSamples(gamma, mvs2, outcome, 2)
		player2.Train(sample2)
		i++
	}
	fmt.Println("1", one, "2", two, "draw", draw)
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

func makeSamples(gamma float64, mvs []*mlann.Matrix, outcome, pid int) *mlann.Sample {
	//compute the reward from the outcome
	reward := float64(0.0)
	if outcome > 0 && outcome == pid {
		//player 1 won
		reward = 10.0
	}
	if outcome > 0 && outcome != pid {
		reward = -10.0
	}
	X := mvs[len(mvs)-1]
	Y := mlann.NewMatrix(1, 1)
	y := reward
	Y.Set(0, 0, y)

	for idx := len(mvs) - 2; idx >= 0; idx-- {
		y = gamma * y
		Y = Y.AppendScalarColumn(y)
		X = X.AppendColumns(mvs[idx])
	}
	//apply the discounted reward to each state action pair observed in the game and
	//update the network based on the results
	return mlann.NewSample(X, Y)
}
