package main

import (
	"flag"
	"fmt"

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

// episode plays a game of tic tac toe asking player1 and then player2 to move on a shared
// board until the game has ended.
func episode(player1, player2 tictactoe.Player) (g *tictactoe.GamePlayed, outcome int) {
	b := tictactoe.NewBoard()
	b.Reset()
	w := 0
	b.Display()
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
		player1.Display(b)
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
	g = b.GamePlayed()
	return
}

func main() {
	flag.Parse()
	if netpath == "" {
		flag.PrintDefaults()
		return
	}

	b := &tictactoe.BoardImp{}
	b.Reset()

	var player1 tictactoe.Player
	var player2 tictactoe.Player

	if player == 1 {
		fmt.Println("I'll be X,You'll be O. I go first")
		fmt.Println("enter your move as: row col")
		player1 = tictactoe.NewMlannPlayer(1, netpath, 0.01, 0.9)
		player2 = tictactoe.NewHumanPlayer(2)
	} else {
		fmt.Println("I'll be O,You'll be X. You go first")
		fmt.Println("enter your move as: row col")
		player1 = tictactoe.NewHumanPlayer(1)
		player2 = tictactoe.NewMlannPlayer(2, netpath, 0.01, 0.9)
	}

	trainplayers(player1, player2, episodes, 0.5)
	if player == 1 {
		player1.Persist(netpath)
	} else {
		player2.Persist(netpath)
	}
}

func trainplayers(player1, player2 tictactoe.Player, episodes int, gamma float64) {
	games := make([]*tictactoe.GamePlayed, 0)
	for i := 0; i < episodes; i++ {
		//play a game and get the sequence of [board,mv] and who won
		g, _ := episode(player1, player2)
		games = append(games, g)
	}
	player1.Train(games)
	player2.Train(games)
}
