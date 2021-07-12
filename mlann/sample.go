package mlann

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

//Sample pairs an input x with an expected output y
type Sample struct {
	x *Matrix
	y *Matrix
}

//NewSample constructs a Sample and returns it
func NewSample(x, y *Matrix) *Sample {
	return &Sample{x: x, y: y}
}

//X gets the input matrix for the sample
func (s *Sample) X() *Matrix {
	return s.x
}

//Y gets the output matrix for the sample
func (s *Sample) Y() *Matrix {
	return s.y
}

func (s *Sample) String() string {
	return fmt.Sprintf("%s\n%s", s.x, s.y)
}

func (s *Sample) Write(w io.Writer) {
	s.x.Write(w)
	w.Write([]byte("\n"))
	s.y.Write(w)
}

//LoadSample reads a sample
func LoadSample(r io.Reader) (*Sample, error) {
	rdr := bufio.NewReader(r)
	data, err := readline(rdr)
	if err != nil {
		return nil, err
	}
	x, err := LoadMatrix(strings.NewReader(data))
	if err != nil {
		return nil, err
	}

	data, err = readline(rdr)
	if err != nil {
		return nil, err
	}

	y, err := LoadMatrix(strings.NewReader(data))
	if err != nil {
		return nil, err
	}

	return &Sample{x: x, y: y}, nil
}

func readline(rdr *bufio.Reader) (string, error) {
	line, prefix, err := rdr.ReadLine()
	if err != nil {
		return "", err
	}
	data := string(line)
	for prefix {
		line, prefix, err = rdr.ReadLine()
		if err != nil {
			return "", err
		}
		data = data + string(line)
	}
	return data, nil
}
