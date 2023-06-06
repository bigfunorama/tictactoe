package main

import (
	"flag"
	"fmt"

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

	// 1. Load two players
	var player1 tictactoe.Player
	var player2 tictactoe.Player

	switch splayer1 {
	case "randoplayer":
		player1 = tictactoe.NewRandomPlayer(1)
	case "mlannplayer":
		if net1path == "" {
			net1path = "player1.net"
		}
		player1 = tictactoe.NewMlannPlayer(1, net1path, epsilon, gamma)
	case "gruplayer":
		if net1path == "" {
			net1path = "player1.net"
		}
		player1 = tictactoe.NewGruPlayer(1, net1path, epsilon)
	}

	switch splayer2 {
	case "randoplayer":
		player2 = tictactoe.NewRandomPlayer(2)
	case "mlannplayer":
		if net2path == "" {
			net2path = "player2.net"
		}
		player2 = tictactoe.NewMlannPlayer(2, net2path, epsilon, gamma)
	}

	// train the two players by having them play each other
	fmt.Println(splayer1, "vs", splayer2)
	trainplayers(player1, player2, episodes, gamma)

	// Persist the results.
	player1.Persist(net1path)
	player2.Persist(net2path)
}

func trainplayers(player1, player2 tictactoe.Player, episodes int, gamma float64) {
	cone := 0
	ctwo := 0
	cdraw := 0
	games := make([]*tictactoe.GamePlayed, 0)
	bsize := 200
	var sone float64
	var stwo float64
	var sdraw float64
	batches := 0
	for i := 0; i < episodes; i++ {
		//play a game and get the sequence of [board,mv] and who won
		g, outcome := episode(player1, player2)

		games = append(games, g)
		switch outcome {
		case 1:
			cone++
		case 2:
			ctwo++
		default:
			cdraw++
		}

		if i > 0 && i%bsize == 0 {
			player1.Train(games)
			player2.Train(games)
			pone := float64(cone) / float64(bsize)
			ptwo := float64(ctwo) / float64(bsize)
			pdraw := float64(cdraw) / float64(bsize)
			sone += pone
			stwo += ptwo
			sdraw += pdraw
			batches += 1

			fmt.Printf("%d, %.2f, %.2f, %.2f\n", i, sone/float64(batches), stwo/float64(batches), sdraw/float64(batches))
			cone = 0
			ctwo = 0
			cdraw = 0
			games = make([]*tictactoe.GamePlayed, 0)
		}
	}
	fmt.Printf("final: %d, one: %.2f, two: %.2f, draw: %.2f\n", batches, sone/float64(batches), stwo/float64(batches), sdraw/float64(batches))
}

// episode plays a game of tic tac toe asking player1 and then player2 to move on a shared
// board until the game has ended.
func episode(player1, player2 tictactoe.Player) (g *tictactoe.GamePlayed, outcome int) {
	b := tictactoe.NewBoard()
	b.Reset()
	w := 0
	for w == 0 {
		mv, err := player1.Move(b)
		if err != nil {
			fmt.Println(err.Error())
			break
		}

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
	g = b.GamePlayed()
	return
}
