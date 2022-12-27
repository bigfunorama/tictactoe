package tictactoe

import (
	"bufio"
	"fmt"
	"os"
	"testing"

	"bigfunbrewing.com/mlann"
)

func loadNetwork(netpath string) (*mlann.Network, error) {
	f, err := os.Open(netpath)
	if err != nil {
		fmt.Println(err.Error())
		return nil, err
	}
	rdr := bufio.NewReader(f)
	net, err := mlann.ReadNetwork(rdr)
	if err != nil {
		fmt.Println(err.Error())
		return nil, err
	}
	return net, nil
}

func TestNetworkValue(t *testing.T) {
	boards := []Board{
		&BoardImp{data: [][]int{
			{0, 0, 0},
			{0, 0, 0},
			{0, 0, 0},
		}},
		&BoardImp{data: [][]int{
			{1, 1, 0},
			{2, 2, 0},
			{0, 0, 0},
		}},
		&BoardImp{data: [][]int{
			{1, 0, 0},
			{0, 1, 0},
			{2, 2, 0},
		}},
	}
	net, err := loadNetwork("./game/player1.net")
	if err != nil {
		t.Fatal(err.Error())
	}
	player := NewMlannPlayer(1, 0.0, net)
	for i := range boards {
		fmt.Println("Board", i)
		boards[i].Display()
		player.Display(boards[i])
	}
}
