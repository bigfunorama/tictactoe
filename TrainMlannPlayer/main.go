package main

import (
	"fmt"

	"bigfunbrewing.com/mlann"
	"bigfunbrewing.com/tictactoe"
)

func main() {
	// make a network
	net := mlann.NewNetwork(
		18,             //inputs 18, 9 for the board, 9 for the move
		0.03,           // alpha the learning_rate
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

	gamma := 0.9

	i := 0
	for i < 1000 {
		//make the players
		mlannplayer := tictactoe.NewMlannPlayer(1, net)
		randoplayer := tictactoe.NewRandomPlayer(2)

		//play a game and get the sequence of [board,mv] and who won
		mvs, outcome := episode(mlannplayer, randoplayer)

		//produce updates
		sample := makeSamples(gamma, mvs, outcome, net)

		fmt.Println("X", sample.X().Shape())
		fmt.Println("y", sample.Y().Shape())
		//Update the network
		net = net.Adam(sample)
		i++
	}

	i = 0
	win := 0
	loss := 0
	draw := 0
	for i < 1000 {
		mlannplayer := tictactoe.NewMlannPlayer(1, net)
		randoplayer := tictactoe.NewRandomPlayer(2)

		//play a game and get the sequence of [board,mv] and who won
		_, outcome := episode(mlannplayer, randoplayer)

		switch outcome {
		case 0:
			draw++
		case 1:
			win++
		case 2:
			loss++
		}
		i++
	}
	fmt.Println("Wins", win)
	fmt.Println("Ties", draw)
	fmt.Println("Losses", loss)
}

func round(b tictactoe.Board, p tictactoe.Player) (int, error) {
	mv, err := p.Move(b)
	if err != nil {
		return -2, err
	}
	err = b.Move(mv)
	if err != nil {
		return -2, err
	}
	return b.GameOver(), nil
}

func episode(mlannplayer, randoplayer tictactoe.Player) ([]*mlann.Matrix, int) {
	b := tictactoe.NewBoard()
	b.Reset()
	w := 0
	mvs := make([]*mlann.Matrix, 0)

	for w == 0 {
		mv, err := mlannplayer.Move(b)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		in := tictactoe.MakeInput(b, mv)
		mvs = append(mvs, in)
		err = b.Move(mv)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		w = b.GameOver()
		if w == -1 || w == 1 {
			b.Display()
			break
		}
		w, err = round(b, randoplayer)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		if w == -1 || w == 2 {
			b.Display()
			break
		}
	}
	if w > 0 {
		fmt.Println("player ", w, "wins!")
	}
	if w == -1 {
		fmt.Println("it's a tie.")
	}
	return mvs, w
}

func makeSamples(gamma float64, mvs []*mlann.Matrix, outcome int, net *mlann.Network) *mlann.Sample {
	//compute the reward from the outcome
	reward := float64(0.0)
	if outcome == 1 {
		//player 1 won
		reward = 10.0
	}
	if outcome == 2 {
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
