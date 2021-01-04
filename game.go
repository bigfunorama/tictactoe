package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/bigfunorama/tictactoe/model"
)

func readMove() (mv *model.Move) {

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
					mv = &model.Move{Pid: 1, Row: row, Col: col}
					return
				}
			}

		}
	}
	return
}

func round(b model.Board, p model.Player) int {
	mv1 := readMove()
	valid := b.Validate(mv1)
	if valid {
		err := b.Move(mv1)
		if err != nil {
			fmt.Println(err)
		}
		mv2, err := p.Move(b)
		if err != nil {
			return 0
		}
		b.Move(mv2)
	} else {
		fmt.Println("Hey, that move won't work")
		return 0
	}
	b.Display()
	return b.GameOver()
}
func main() {

	fmt.Println("You be X, I'll be O. You go first")
	fmt.Println("enter your move as: row col")
	b := &model.BoardImp{}
	b.Reset()
	b.Display()
	player2 := model.NewRandomPlayer(2)
	done := round(b, player2)
	for done == 0 {
		done = round(b, player2)
	}
}
