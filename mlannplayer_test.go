package tictactoe

import (
	"fmt"
	"testing"
)

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
	player := NewMlannPlayer(1, "./game/player1.net", 0.0, 0.9)
	for i := range boards {
		fmt.Println("Board", i)
		boards[i].Display()
		player.Display(boards[i])
	}
}
