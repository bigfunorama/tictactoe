package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"bigfunbrewing.com/mlann"
	"bigfunbrewing.com/tictactoe"
)

var netpath string
var player int

func init() {
	flag.StringVar(&netpath, "netpath", "", "path to the network to play against")
	flag.IntVar(&player, "player", 1, "which player is the network. Default is 1")
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
	f, err := os.Open(netpath)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	rdr := bufio.NewReader(f)
	net, err := mlann.ReadNetwork(rdr)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	fmt.Println("I'll be X,You'll be O. I go first")
	fmt.Println("enter your move as: row col")
	b := &tictactoe.BoardImp{}
	b.Reset()

	player1 := tictactoe.NewMlannPlayer(1, 0.0, net)
	player2 := tictactoe.NewHumanPlayer(2)
	episode(player1, player2)
}
