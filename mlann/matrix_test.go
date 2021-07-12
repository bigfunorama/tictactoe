package mlann

import (
	"fmt"
	"strings"
	"testing"
)

type testSetValue struct {
	r        int
	c        int
	val      float64
	expected error
}

func TestLoadMatrix(t *testing.T) {
	mtx, err := LoadMatrix(strings.NewReader("3,4,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0"))
	if err != nil {
		t.Error(err.Error())
	}
	if mtx.Rows() != 3 {
		t.Errorf("expected 3 rows, got %d", mtx.Rows())
	}

	if mtx.Cols() != 4 {
		t.Errorf("expected 4 columns, got %d", mtx.Cols())
	}

	for idx, item := range mtx.data {
		if item != float64(idx+1) {
			t.Errorf("expected %d, got %.5f", idx+1, item)
		}
	}
}

func TestWriteMatrix(t *testing.T) {
	mtx, err := LoadMatrix(strings.NewReader("3,4,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0"))
	if err != nil {
		t.Error(err.Error())
	}
	w := &strings.Builder{}
	mtx.Write(w)
	mtx2, err := LoadMatrix(strings.NewReader(w.String()))
	if mtx2.Rows() != mtx.Rows() {
		t.Errorf("expected %d, got %d", mtx.Rows(), mtx2.Rows())
	}

	if mtx2.Cols() != mtx.Cols() {
		t.Errorf("expected %d, got %d", mtx.Cols(), mtx2.Cols())
	}
	for idx, item := range mtx2.data {
		if item != mtx.data[idx] {
			t.Errorf("expected %.5f, got %.5f", mtx.data[idx], item)
		}
	}
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
			t.Errorf("error, expected 1.0, got %.2f", v)
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

func TestMul2(t *testing.T) {
	a := &Matrix{r: 2, c: 2, data: []float64{1, 2,
		3, 4}}
	b := &Matrix{r: 2, c: 3, data: []float64{1, 2, 3,
		4, 5, 6}}
	out, err := a.Mul(b)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6,
		3*1 + 4*4, 3*2 + 4*5, 3*3 + 4*6}
	for i := 0; i < len(output); i++ {
		if output[i] != out.data[i] {
			t.Errorf("incorrect value, got %.2f, expected %.2f", out.data[i], output[i])
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
			t.Errorf("incorrect value, got %.2f, expected %.2f", e.data[i], output[i])
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
			t.Errorf("incorrect value, got %.2f, expected %.2f", e.data[i], output[i])
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
			t.Errorf("incorrect value, got %.2f, expected %.2f", e.data[i], output[i])
		}
	}
}

func TestTranspose(t *testing.T) {
	m := &Matrix{r: 2, c: 2, data: []float64{1, 2, 3, 4}}
	out := m.Transpose()
	output := []float64{1, 3, 2, 4}
	for i := 0; i < len(output); i++ {
		if output[i] != out.data[i] {
			t.Errorf("incorrect value, got %.2f, expected %.2f", out.data[i], output[i])
		}
	}
}

func TestAppendColumns(t *testing.T) {
	m := Identity(4)
	out, err := m.AppendColumns(m)
	if err != nil {
		t.Error(err.Error())
	}
	output := []float64{1, 0, 0, 0, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 0, 1, 0,
		0, 0, 0, 1, 0, 0, 0, 1}
	for i := 0; i < len(output); i++ {
		if output[i] != out.data[i] {
			t.Errorf("incorrect value, got %.2f, expected %.2f", out.data[i], output[i])
		}
	}
}

func TestAppendRows(t *testing.T) {
	m := Identity(4)
	out, err := m.AppendRows(m)
	if err != nil {
		t.Error(err.Error())
	}
	output := []float64{1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	for i := 0; i < len(output); i++ {
		if output[i] != out.data[i] {
			t.Errorf("incorrect value, got %.2f, expected %.2f", out.data[i], output[i])
		}
	}
}

func TestAppendColumns2(t *testing.T) {
	a := &Matrix{r: 2, c: 6, data: []float64{4, 2, -1, 0, 1, 0, 1, 4, 0, -1, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	tmp, err := a.AppendColumns(b)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{4, 2, -1, 0, 1, 0, 12, 1, 4, 0, -1, 0, 1, 6}
	for i := 0; i < len(output); i++ {
		if tmp.data[i] != output[i] {
			t.Errorf("expected %.5f, got %.5f", output[i], tmp.data[i])
		}
	}
}

func TestAppendColumns3(t *testing.T) {
	c := &Matrix{r: 1, c: 6, data: []float64{1, 2, 3, 4, 5, 6}}
	tmp, err := c.AppendColumns(&Matrix{c: 1, r: 1, data: []float64{7}})
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{1, 2, 3, 4, 5, 6, 7}
	for i := 0; i < len(output); i++ {
		if tmp.data[i] != output[i] {
			t.Errorf("expected %.5f, got %.5f", output[i], tmp.data[i])
		}
	}
}

func TestMultipleAppends(t *testing.T) {
	a := &Matrix{r: 2, c: 6, data: []float64{4, 2, -1, 0, 1, 0, 1, 4, 0, -1, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	tmp, _ := a.AppendColumns(b)
	fmt.Println(tmp)
	c := &Matrix{r: 1, c: 6, data: []float64{1, 2, 3, 4, 5, 6}}
	tmp2, _ := c.AppendColumns(&Matrix{c: 1, r: 1, data: []float64{7}})
	fmt.Println(tmp2)
	out, err := tmp.AppendRows(tmp2)
	if err != nil {
		t.Error(err)
		return
	}
	fmt.Println(out)
	output := []float64{4, 2, -1, 0, 1, 0, 12, 1, 4, 0, -1, 0, 1, 6, 1, 2, 3, 4, 5, 6, 7}
	for i := 0; i < len(output); i++ {
		if out.data[i] != output[i] {
			t.Errorf("expected %.5f, got %.5f", output[i], out.data[i])
		}
	}
}

func TestExtractMatrix(t *testing.T) {
	m := Identity(4)
	out, err := m.ExtractMatrix(2, 2, 4, 4)
	if err != nil {
		t.Error(err.Error())
	}
	output := []float64{1, 0, 0, 1}
	for i := 0; i < len(output); i++ {
		if output[i] != out.data[i] {
			t.Errorf("incorrect value, got %.2f, expected %.2f", out.data[i], output[i])
		}
	}
}

func TestInverse(t *testing.T) {
	m := &Matrix{r: 4, c: 4, data: []float64{2, 5, 10, 0,
		1, 1, 1, 0,
		-2, -10, -30, 1,
		-1, -2, -3, 0}}
	out, err := m.Inverse()
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{0.5, 2.5, 0, 2.5, -1, -2, 0, -4, 0.5, 0.5, 0, 1.5, 6, 0, 1, 10}
	for i := 0; i < len(output); i++ {
		if out.data[i] < (output[i]-precision) || out.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, got %.5f, expected %.5f", out.data[i], output[i])
		}
	}
}

func TestInverseSingular(t *testing.T) {
	m := &Matrix{r: 4, c: 4, data: []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
	_, err := m.Inverse()
	if err == nil {
		t.Error("Matrix is singular and inverse should return an error")
		return
	}
}

func TestPivot(t *testing.T) {
	m := &Matrix{r: 4, c: 4, data: []float64{2, 5, 10, 0, 1, 1, 1, 0, -2, -10, -30, 1, -1, -2, -3, 0}}
	out, err := m.Pivot(0, 0)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{1, 2.5, 5, 0,
		0, -1.5, -4, 0,
		0, -5, -20, 1,
		0, 0.5, 2, 0,
	}
	for i := 0; i < len(output); i++ {
		if out.data[i] < (output[i]-precision) || out.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, got %.5f, expected %.5f", out.data[i], output[i])
		}
	}
}

func TestLeastSquares(t *testing.T) {
	a := &Matrix{r: 3, c: 2, data: []float64{0.3, 0.1, 0.4, 0.2, 0.3, 0.7}}
	b := &Matrix{r: 3, c: 1, data: []float64{5, 3, 4}}
	pinv, err := a.PseudoInverse()
	if err != nil {
		t.Error(err)
		return
	}
	ans, err := pinv.Mul(b)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{10.566502463, 0.960591133}
	for i := 0; i < len(output); i++ {
		if ans.data[i] < (output[i]-precision) ||
			ans.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.9f, got %.9f", output[i], ans.data[i])
		}
	}
}

func TestHadamard(t *testing.T) {
	lt := &Matrix{r: 4, c: 4, data: []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
	ans, err := lt.Hadamard(lt)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	for i := 0; i < len(output); i++ {
		if ans.data[i] < (output[i]-precision) ||
			ans.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.9f, got %.9f", output[i], ans.data[i])
		}
	}
}

func TestApply(t *testing.T) {
	lt := &Matrix{r: 6, c: 1, data: []float64{-2, -1, 0, 1, 2, 3}}
	ans := lt.Apply(&RELU{})
	output := []float64{0, 0, 0, 1, 2, 3}
	for i := 0; i < len(output); i++ {
		if ans.data[i] < (output[i]-precision) ||
			ans.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.9f, got %.9f", output[i], ans.data[i])
		}
	}
}
