package mlann

import (
	"fmt"
	"strings"
	"testing"
)

type testWriter struct{}

func TestWrite(t *testing.T) {
	x, err := LoadMatrix(strings.NewReader("2,2,1.0,2.0,3.0,4.0"))
	if err != nil {
		t.Error(err.Error())
	}
	y, err := LoadMatrix(strings.NewReader("2,1,1.0,0.0"))
	if err != nil {
		t.Error(err.Error())
	}
	s := &Sample{x: x, y: y}
	bb := &strings.Builder{}
	s.Write(bb)
	fmt.Println(bb.String())
	sample, err := LoadSample(strings.NewReader(bb.String()))
	if err != nil {
		t.Error(err.Error())
	}
	if !s.x.Equals(sample.x) {
		t.Error("x matrices are not equal")
	}

	if !s.y.Equals(sample.y) {
		t.Error("y matrices are not equal")
	}
}
