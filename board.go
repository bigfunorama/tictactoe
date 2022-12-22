package tictactoe

import (
	"fmt"
)

// Board provides a means of displaying the current game state:w
type Board interface {
	Display()
	Validate(mv *Move) bool
	Move(mv *Move) error
	GameOver() int
	Get(r, c int) (int, error)
	Reset()
}

// NewBoard returns an instance of a tic tac toe board for play.
func NewBoard() Board {
	b := &BoardImp{}
	return b
}

// BoardImp is an implementation of a tictactoe board
type BoardImp struct {
	data [][]int
}

// Get is an accessor for a board position and returns
// the value of the board at position row, col
func (b *BoardImp) Get(row, col int) (int, error) {
	if row < 0 || row > 2 {
		return -1, fmt.Errorf("invalid row")
	}
	if col < 0 || col > 2 {
		return -1, fmt.Errorf("invalid column")
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

func dplayer(player int) (ps string) {
	ps = " "
	if player == 1 {
		ps = "X"
	}
	if player == 2 {
		ps = "O"
	}
	return
}

// Display produces a representation of the current Board state
func (b *BoardImp) Display() {
	fmt.Println("   | 0 | 1 | 2 ")
	fmt.Println("---+---+---+---")
	for i := range b.data {
		fmt.Printf(" %d | %s | %s | %s\n", i, dplayer(b.data[i][0]), dplayer(b.data[i][1]), dplayer(b.data[i][2]))
		if i < 2 {
			fmt.Println("---+---+---+---")
		} else {
			fmt.Println()
		}
	}
}

// Validate validates whether a move to position row and col
// by player is allowed given the current board state.
func (b *BoardImp) Validate(mv *Move) bool {
	row := mv.Row
	col := mv.Col
	if row < 0 || row > 2 {
		return false
	}
	if col < 0 || col > 2 {
		return false
	}
	if b.data[row][col] == 0 {
		return true
	}
	return false
}

type InvalidPositionError struct {
	row int
}

func (e *InvalidPositionError) Error() string {
	return fmt.Sprintf("index %d out of bounds must be in [0,2]", e.row)
}

type NonEmptyPositionError struct {
	row, col int
	player   int
}

func (e *NonEmptyPositionError) Error() string {
	return fmt.Sprintf("position (%d,%d) occupied by player %d", e.row, e.col, e.player)
}

// Move attempts to assign player to position row,col. It
// returns an error if the position is invalid or the player
// cannot move there
func (b *BoardImp) Move(mv *Move) error {
	row := mv.Row
	col := mv.Col
	player := mv.Pid
	if row < 0 || row > 2 {
		return &InvalidPositionError{row: row}
	}
	if col < 0 || col > 2 {
		return &InvalidPositionError{row: col}
	}
	if b.data[row][col] != 0 {
		return &NonEmptyPositionError{row: row, col: col, player: b.data[row][col]}
	}
	if player == 1 || player == 2 {
		fmt.Printf("player %d moved to (%d,%d)\n", player, row, col)
		b.data[row][col] = player
		return nil
	}
	return fmt.Errorf("invalid player")
}

// GameOver determines whether the game is over.
// 0 implies that the game is not over
// 1 implies that the game is over and player 1 won
// 2 implies that the game is over and player 2 won
// -1 implies that the game is over and it's a tie
func (b *BoardImp) GameOver() int {
	//Verify whether the game is finished or not
	//1. three in a row of any player
	//1.a. across
	for i := range b.data {
		if b.data[i][0] == b.data[i][1] &&
			b.data[i][0] == b.data[i][2] &&
			b.data[i][0] != 0 {
			return b.data[i][0]
		}
	}
	//1.b. down
	for j := range b.data[0] {
		if b.data[0][j] == b.data[1][j] &&
			b.data[0][j] == b.data[2][j] &&
			b.data[0][j] != 0 {
			return b.data[0][j]
		}
	}
	//1.c. diaganol
	if b.data[0][0] == b.data[1][1] &&
		b.data[0][0] == b.data[2][2] &&
		b.data[0][0] != 0 {
		return b.data[0][0]
	}
	if b.data[0][2] == b.data[1][1] &&
		b.data[0][2] == b.data[2][0] &&
		b.data[0][2] != 0 {
		return b.data[0][2]
	}
	//2. no three in a row but no empty spaces
	empty := 0
	for i := range b.data {
		for j := range b.data[i] {
			if b.data[i][j] == 0 {
				empty++
			}
		}
	}
	if empty == 0 {
		return -1
	}
	return 0
}
