package tictactoe

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"bigfunbrewing.com/mlann"
)

type Player interface {
	Move(b Board) (mv *Move, err error)
	Train(sample *mlann.Sample)
}

type RandomPlayer struct {
	pid int
}

type Move struct {
	Pid int
	Row int
	Col int
}

type GameOver struct{}

func (g *GameOver) Error() string {
	return "No more moves to make. Game Over."
}

func NewRandomPlayer(pid int) *RandomPlayer {
	return &RandomPlayer{pid: pid}
}

func (rp *RandomPlayer) Move(b Board) (mv *Move, err error) {
	moves, err := ValidMoves(b, rp.pid)
	if err != nil {
		return nil, err
	}
	idx := rand.Intn(len(moves))
	mv = moves[idx]
	return
}

func (rp *RandomPlayer) Train(sample *mlann.Sample) {
	//do nothing
}

func ValidMoves(b Board, pid int) ([]*Move, error) {
	moves := make([]*Move, 0)
	for rr := 0; rr < 3; rr++ {
		for cc := 0; cc < 3; cc++ {
			mv := &Move{Pid: pid, Row: rr, Col: cc}
			if b.Validate(mv) {
				moves = append(moves, mv)
			}
		}
	}
	if len(moves) == 0 {
		return nil, &GameOver{}
	}
	return moves, nil
}

type HumanPlayer struct {
	pid int
}

func NewHumanPlayer(pid int) *HumanPlayer {
	return &HumanPlayer{pid: pid}
}

func readMove(pid int) (mv *Move) {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("row col: ")
		text, _ := reader.ReadString('\n')

		items := strings.Split(text, " ")
		if len(items) != 2 {
			fmt.Println("Error reading your input. remember, row col")
		} else {
			item := strings.Trim(items[0], " \r\n")
			fmt.Printf("parsed row |%s|\n", item)
			row, err := strconv.Atoi(item)
			if err != nil {
				fmt.Println("Error reading your input. remember, row, col are numbers both in {0,1,2}")
			} else {
				item := strings.Trim(items[1], " \r\n")
				fmt.Printf("parsed col |%s|\n", item)
				col, err := strconv.Atoi(item)
				if err != nil {
					fmt.Println("Error reading your input. remember, row, col are numbers both in {0,1,2}")
				} else {
					mv = &Move{Pid: pid, Row: row, Col: col}
					return
				}
			}

		}
	}
}

func (hp *HumanPlayer) Move(b Board) (mv *Move, err error) {
	mv = readMove(hp.pid)
	return
}

func (hp *HumanPlayer) Train(sample *mlann.Sample) {

}
