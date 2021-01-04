package matrix

import "testing"

type testSetValue struct {
	r        int
	c        int
	val      float64
	expected error
}

func TestDimensions(t *testing.T) {
	m := NewMatrix(3, 4)
	if m.Rows() != 3 {
		t.Errorf("error fetching rows, expected 3, got %d", m.Rows())
	}
	if m.Cols() != 4 {
		t.Errorf("error fetching cols, expected 3, got %d", m.Cols())
	}
}

func TestMatrixSet(t *testing.T) {
	tests := []testSetValue{
		{r: -1, c: 0, val: 1.0, expected: &InvalidRowError{row: -1}},
		{r: 0, c: -1, val: 1.0, expected: &InvalidColumnError{col: -1}},
		{r: 0, c: 0, val: 1.0, expected: nil},
	}
	for _, t2 := range tests {
		m := NewMatrix(3, 3)
		e := m.Set(t2.r, t2.c, t2.val)
		if e == nil && !(t2.expected == nil) {
			t.Errorf("error, got nil but expected '%s'", t2.expected.Error())
		} else {
			if e.Error() != t2.expected.Error() {
				t.Errorf("error, expected '%s', got '%s'", t2.expected, e)
			}
		}
	}
}

func TestMatrixGet(t *testing.T) {
	tests := []testSetValue{
		{r: 0, c: 0, val: 1.0, expected: nil},
		{r: 1, c: 1, val: 1.0, expected: nil},
		{r: 2, c: 2, val: 1.0, expected: nil},
	}
	m := NewMatrix(3, 3)
	for _, t2 := range tests {
		m.Set(t2.r, t2.c, t2.val)
	}

	for _, t3 := range tests {
		v, err := m.Get(t3.r, t3.c)
		if err != nil {
			t.Error(err)
		}
		if t3.val != v {
			t.Errorf("got the wrong value, expected %f, got %f", t3.val, v)
		}
	}
}

func TestIdentity(t *testing.T) {
	I := Identity(3)
	for i := 0; i < I.Rows(); i++ {
		v, _ := I.Get(i, i)
		if v != 1.0 {
			t.Errorf("error, expected 1.0, got %f", v)
		}
	}
}

func TestEquals(t *testing.T) {
	lt := Identity(4)
	rt := Identity(4)
	if !(lt.Equals(rt)) {
		t.Errorf("error these two Identity matrices should be equal")
	}
}

func TestClone(t *testing.T) {
	I := Identity(3)
	n := I.Clone()
	if !(I.Equals(n)) {
		t.Error("a clone should equal the original")
	}
}

func TestRow(t *testing.T) {
	I := Identity(3)
	row0, err := I.Row(0)
	if err != nil {
		t.Error(err.Error())
	}
	exp := &Matrix{r: 1, c: 3, data: []float64{1.0, 0.0, 0.0}}
	if !(row0.Equals(exp)) {
		t.Errorf("row 0 of the identity was not as expected, got %s", row0)
	}
}

func TestCol(t *testing.T) {
	I := Identity(3)
	col0, err := I.Col(0)
	if err != nil {
		t.Error(err.Error())
	}
	exp := &Matrix{r: 3, c: 1, data: []float64{1.0, 0.0, 0.0}}
	if !(col0.Equals(exp)) {
		t.Errorf("col 0 of the identity matrix was not as expected, got %s", col0)
	}
}

type testadd struct {
	lt *Matrix
	rt *Matrix
	eq *Matrix
	e  error
}

func TestMatrixAddition(t *testing.T) {
	tests := []testadd{
		{
			lt: &Matrix{r: 2, c: 2, data: []float64{1, 1, 1, 1}},
			rt: &Matrix{r: 2, c: 2, data: []float64{1, 2, 3, 4}},
			eq: &Matrix{r: 2, c: 2, data: []float64{2, 3, 4, 5}},
			e:  nil,
		},
		{
			lt: &Matrix{r: 1, c: 4, data: []float64{1, 1, 1, 1}},
			rt: &Matrix{r: 2, c: 2, data: []float64{1, 2, 3, 4}},
			eq: nil,
			e:  &InvalidAddOperation{},
		},
	}
	for _, t2 := range tests {
		out, err := t2.lt.Add(t2.rt)
		if err == nil {
			if t2.e != nil {
				t.Errorf("no error returned but expected %s", t2.e.Error())
			} else {
				if !out.Equals(t2.eq) {
					t.Errorf("error, got %s, expected %s", out, t2.eq)
				}
			}
		} else {
			if t2.e == nil {
				t.Error(err.Error())
			}
		}
	}
}

func TestMatrixMultiplication(t *testing.T) {
	tests := []testadd{
		{
			lt: &Matrix{r: 2, c: 2, data: []float64{1, 1, 1, 1}},
			rt: &Matrix{r: 2, c: 2, data: []float64{2, 2, 2, 2}},
			eq: &Matrix{r: 2, c: 2, data: []float64{4, 4, 4, 4}},
			e:  nil,
		},
		{
			lt: &Matrix{r: 1, c: 4, data: []float64{1, 1, 1, 1}},
			rt: &Matrix{r: 4, c: 1, data: []float64{1, 2, 3, 4}},
			eq: &Matrix{r: 1, c: 1, data: []float64{10}},
			e:  nil,
		},
		{
			lt: &Matrix{r: 4, c: 1, data: []float64{1, 1, 1, 1}},
			rt: &Matrix{r: 1, c: 4, data: []float64{1, 2, 3, 4}},
			eq: &Matrix{r: 4, c: 4, data: []float64{1, 2, 3, 4,
				1, 2, 3, 4,
				1, 2, 3, 4,
				1, 2, 3, 4,
			}},
			e: nil,
		},
	}
	for _, t2 := range tests {
		out, err := t2.lt.Mul(t2.rt)
		if err == nil {
			if t2.e != nil {
				t.Errorf("no error returned but expected %s", t2.e.Error())
			} else {
				if !out.Equals(t2.eq) {
					t.Errorf("error, got %s, expected %s", out, t2.eq)
				}
			}
		} else {
			if t2.e == nil {
				t.Error(err.Error())
			}
		}
	}
}

func TestScalarRowMultiply(t *testing.T) {
	I := Identity(3)
	e, err := I.RowScalarMultiply(1, 3.0)
	if err != nil {
		t.Error(err.Error())
	}
	output := []float64{1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0}
	for i := 0; i < len(output); i++ {
		if output[i] != e.data[i] {
			t.Errorf("incorrect value, got %f2, expected %f2", e.data[i], output[i])
		}
	}
}

func TestRowInterchange(t *testing.T) {
	I := Identity(3)
	e, err := I.RowInterchange(0, 2)
	if err != nil {
		t.Error(err.Error())
	}
	output := []float64{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0}
	for i := 0; i < len(output); i++ {
		if output[i] != e.data[i] {
			t.Errorf("incorrect value, got %f2, expected %f2", e.data[i], output[i])
		}
	}
}

func TestRowMulAdd(t *testing.T) {
	I := Identity(3)
	e, err := I.RowMulAdd(0, 3.0, 2)
	if err != nil {
		t.Error(err.Error())
		return
	}
	output := []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0}
	for i := 0; i < len(output); i++ {
		if output[i] != e.data[i] {
			t.Errorf("incorrect value, got %f2, expected %f2", e.data[i], output[i])
		}
	}
}

func TestTranspose(t *testing.T) {
	m := &Matrix{r: 2, c: 2, data: []float64{1, 2, 3, 4}}
	out := m.Transpose()
	output := []float64{1, 3, 2, 4}
	for i := 0; i < len(output); i++ {
		if output[i] != out.data[i] {
			t.Errorf("incorrect value, got %f2, expected %f2", out.data[i], output[i])
		}
	}
}
