package main

import (
	"fmt"

	"github.com/bigfunorama/tictactoe/model"
)

func main() {

	fmt.Println("Hello! Want to play a game?")
	b := model.BoardImp{}
	(&b).Reset()
	(&b).Display()
}
