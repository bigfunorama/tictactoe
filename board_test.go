package tictactoe

import (
	"testing"
)

func TestDisplay(t *testing.T) {
	b := NewBoard()
	b.Reset()
	b.Display()
}

func TestReset(t *testing.T) {
	b := BoardImp{}
	b.Reset()
	b.Display()
	b.Move(&Move{1, 0, 0})
}

func TestValidateDoubleMove(t *testing.T) {
	b := BoardImp{}
	b.Reset()
	err := b.Move(&Move{1, 0, 0})
	if err != nil {
		t.Errorf(err.Error())
	}
	err = b.Move(&Move{1, 0, 0})
	if err == nil {
		t.Errorf(err.Error())
	}
}

func TestValidateBadMove(t *testing.T) {
	b := BoardImp{}
	b.Reset()
	err := b.Move(&Move{0, 0, 0})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 0, 3})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 0, -1})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, -1, 0})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 3, 0})
	if err == nil {
		t.Errorf(err.Error())
	}

	err = b.Move(&Move{1, 3, 3})
	if err == nil {
		t.Errorf(err.Error())
	}
}
