package model

import (
	"fmt"
)

//Board provides a means of displaying the current game state:w
type Board interface {
	Display()
	Validate(player, row, col int) bool
	Move(player, row, col int) error
	Done() bool
	Reset()
}

//BoardImp is an implementation of a tictactoe board
type BoardImp struct {
	data [][]int
}

// Get is an accessor for a board position and returns
// the value of the board at position row, col
func (b *BoardImp) get(row, col int) (int, error) {
	if row < 0 || row > 2 {
		return -1, fmt.Errorf("Invalid row")
	}
	if col < 0 || col > 2 {
		return -1, fmt.Errorf("Invalid column")
	}
	return b.data[row][col], nil
}

// Reset sets the board back to the start state.
func (b *BoardImp) Reset() {
	b.data = make([][]int, 3)
	for i := 0; i < 3; i++ {
		b.data[i] = make([]int, 3)
		for j := 0; j < 3; j++ {
			b.data[i][j] = 0
		}
	}
}

//Display produces a representation of the current Board state
func (b *BoardImp) Display() {
	for i := range b.data {
		fmt.Printf(" %d | %d | %d\n", b.data[i][0], b.data[i][1], b.data[i][2])
		if i < 2 {
			fmt.Println("---+---+---")
		} else {
			fmt.Println()
		}
	}
}

// Validate validates whether a move to position row and col
// by player is allowed given the current board state.
func (b *BoardImp) Validate(player, row, col int) bool {
	if row >= 0 || row <= 2 {
		return true
	}
	if col >= 0 || col <= 2 {
		return true
	}
	if b.data[row][col] == 0 {
		return true
	}
	return false
}

// Move attempts to assign player to position row,col. It
// returns an error if the position is invalid or the player
// cannot move there
func (b *BoardImp) Move(player, row, col int) error {
	if row < 0 || row > 2 {
		return fmt.Errorf("Invalid row")
	}
	if col < 0 || col > 2 {
		return fmt.Errorf("Invalid column")
	}
	if b.data[row][col] != 0 {
		return fmt.Errorf("Position is not empty")
	}
	if player == 1 || player == 2 {
		b.data[row][col] = player
		return nil
	}
	return fmt.Errorf("Invalid player")
}

//Done determines whether the game is over.
func (b *BoardImp) Done() bool {
	return false
}
