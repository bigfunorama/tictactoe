package main

import (
	"flag"
	"fmt"

	"bigfunbrewing.com/tictactoe"
)

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
	for i := range g.Positions() {
		fmt.Println(g.Positions()[i])
	}
	return
}

func main() {
	net1path := flag.String("net1", "", "path to the network to play against")
	net2path := flag.String("net2", "", "path to the network to play against")
	splayer1 := flag.String("player1", "", "choose the type for player 1 (randoplayer|mlannplayer|gruplayer)")
	splayer2 := flag.String("player2", "", "choose the type for player 2 (randoplayer|mlannplayer|gruplayer)")
	episodes := flag.Int("games", 10, "number of games to play")
	gamma := flag.Float64("gamma", 0.9, "gamma is the discount rate on future rewards")
	epsilon := flag.Float64("epsilon", 0.01, "epsilon is the exploration rate for NN players")
	flag.Parse()

	fmt.Println(*net1path, *net2path, *splayer1, *splayer2, *episodes, *gamma, *epsilon)
	if *splayer1 == "" || *splayer2 == "" {
		flag.PrintDefaults()
		return
	}
	if (*splayer1 == "gruplayer" || *splayer1 == "mlannplayer") && *net1path == "" {
		flag.PrintDefaults()
		return
	}
	if (*splayer2 == "gruplayer" || *splayer2 == "mlannplayer") && *net2path == "" {
		flag.PrintDefaults()
		return
	}

	b := &tictactoe.BoardImp{}
	b.Reset()

	var player1 tictactoe.Player
	var player2 tictactoe.Player

	switch *splayer1 {
	case "randoplayer":
		player1 = tictactoe.NewRandomPlayer(1)
	case "mlannplayer":
		player1 = tictactoe.NewMlannPlayer(1, *net1path, *epsilon, *gamma)
	case "gruplayer":
		player1 = tictactoe.NewGruPlayer(1, *net1path, *epsilon)
	case "humanplayer":
		player1 = tictactoe.NewHumanPlayer(1)
	}

	switch *splayer2 {
	case "randoplayer":
		player2 = tictactoe.NewRandomPlayer(2)
	case "mlannplayer":
		player2 = tictactoe.NewMlannPlayer(2, *net2path, *epsilon, *gamma)
	case "gruplayer":
		player2 = tictactoe.NewGruPlayer(2, *net2path, *epsilon)
	case "humanplayer":
		player2 = tictactoe.NewHumanPlayer(2)
	}

	trainplayers(player1, player2, *episodes, 0.9)
	player1.Persist(*net1path)
	player2.Persist(*net2path)
}

func trainplayers(player1, player2 tictactoe.Player, episodes int, gamma float64) {
	games := make([]*tictactoe.GamePlayed, 0)
	for i := 0; i < episodes; i++ {
		//play a game and get the sequence of [board,mv] and who won
		g, _ := episode(player1, player2)
		games = append(games, g)
		player1.Train(games)
		player2.Train(games)
	}
}
