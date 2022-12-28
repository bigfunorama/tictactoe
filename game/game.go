package main

import (
	"flag"
	"fmt"
	"os"

	"bigfunbrewing.com/mlann"
	"bigfunbrewing.com/tictactoe"
)

var netpath string
var player int
var episodes int

func init() {
	flag.StringVar(&netpath, "netpath", "", "path to the network to play against")
	flag.IntVar(&player, "player", 1, "which player is the network. Default is 1")
	flag.IntVar(&episodes, "games", 50, "number of games to play")
}

func episode(player1, player2 tictactoe.Player) (p1mvs, p2mvs []*mlann.Matrix, outcome int) {
	b := tictactoe.NewBoard()
	b.Reset()
	b.Display()
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

		player1.Display(b)
		err = b.Move(mv)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		b.Display()
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

		player2.Display(b)
		err = b.Move(mv)
		if err != nil {
			fmt.Println(err.Error())
			break
		}
		b.Display()
		w = b.GameOver()
		if w == -1 || w == 2 {
			break
		}
	}
	outcome = w
	return
}

func main() {
	flag.Parse()
	if netpath == "" {
		flag.PrintDefaults()
		return
	}
	net, err := mlann.LoadNetworkFromFile(netpath)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	b := &tictactoe.BoardImp{}
	b.Reset()

	var player1 tictactoe.Player
	var player2 tictactoe.Player

	if player == 1 {
		fmt.Println("I'll be X,You'll be O. I go first")
		fmt.Println("enter your move as: row col")
		player1 = tictactoe.NewMlannPlayer(1, 0.01, net)
		player2 = tictactoe.NewHumanPlayer(2)
	} else {
		fmt.Println("I'll be O,You'll be X. You go first")
		fmt.Println("enter your move as: row col")
		player1 = tictactoe.NewHumanPlayer(1)
		player2 = tictactoe.NewMlannPlayer(2, 0.01, net)
	}

	trainplayers(player1, player2, episodes, 0.5)
	f2, err := os.Create(netpath)
	if err != nil {
		fmt.Println("player 1 failed to write", err.Error())
	} else {
		defer f2.Close()
		net.Write(f2)
	}

}

func trainplayers(player1, player2 tictactoe.Player, episodes int, gamma float64) {
	i := 0
	one := 0
	two := 0
	draw := 0
	cone := 0
	ctwo := 0
	cdraw := 0
	for i < episodes {
		//play a game and get the sequence of [board,mv] and who won
		mvs1, mvs2, outcome := episode(player1, player2)
		switch outcome {
		case 1:
			cone++
		case 2:
			ctwo++
		default:
			cdraw++
		}
		if i > 0 && i%200 == 0 {
			one += cone
			two += ctwo
			draw += cdraw
			fmt.Printf("%d, %.2f, %.2f, %.2f\n", i, float64(cone)/float64(200), float64(ctwo)/float64(200), float64(cdraw)/float64(200))
			cone = 0
			ctwo = 0
			cdraw = 0
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

func makeSamples(gamma float64, mvs []*mlann.Matrix, outcome, pid int) *mlann.Sample {
	//compute the reward from the outcome
	reward := float64(0.0)
	if outcome > 0 && outcome == pid {
		//player 1 won
		reward = 1.5
	}
	if outcome > 0 && outcome != pid {
		reward = -0.50
	}
	if outcome == -1 {
		reward = 1.5
	}
	X := mvs[len(mvs)-1]
	Y := mlann.NewMatrix(1, 1)
	y := reward
	Y.Set(0, 0, y)

	for idx := len(mvs) - 2; idx >= 0; idx-- {
		y = gamma*y - 0.10
		Y = Y.AppendScalarColumn(y)
		X = X.AppendColumns(mvs[idx])
	}
	//apply the discounted reward to each state action pair observed in the game and
	//update the network based on the results
	return mlann.NewSample(X, Y)
}
