package matrix

import "fmt"

type Matrix struct {
	data []float64
	r    int
	c    int
}

type InvalidRowError struct {
	row int
}

func (e *InvalidRowError) Error() string {
	return fmt.Sprintf("index %d out of bounds.", e.row)
}

type InvalidColumnError struct {
	col int
}

func (e *InvalidColumnError) Error() string {
	return fmt.Sprintf("index %d out of bounds.", e.col)
}

//NewMatrix constructs a pointer to a Matrix
func NewMatrix(rows, cols int) *Matrix {
	out := Matrix{}
	out.c = cols
	out.r = rows
	out.data = make([]float64, out.c*out.r)
	return &out
}

//Identity constructs an instance of the Identity matrix with r rows and columns
func Identity(r int) *Matrix {
	out := NewMatrix(r, r)
	for i := 0; i < r; i++ {
		out.Set(i, i, 1.0)
	}
	return out
}

//Rows returns the number of rows in this matrix
func (m *Matrix) Rows() int {
	return m.r
}

//Cols returns the number of columns in this matrix
func (m *Matrix) Cols() int {
	return m.c
}

//Row constructs a new matrix from the ith row of this matrix and returns it
func (m *Matrix) Row(i int) (*Matrix, error) {
	if i < 0 || i >= m.r {
		return nil, &InvalidRowError{row: i}
	}
	out := NewMatrix(1, m.c)
	for j := 0; j < m.c; j++ {
		out.data[j] = m.data[i*m.c+j]
	}
	return out, nil
}

//Col constructs a new matrix from the jth column of this matrix and returns it
func (m *Matrix) Col(j int) (*Matrix, error) {
	if j < 0 || j >= m.c {
		return nil, &InvalidColumnError{col: j}
	}
	out := NewMatrix(m.r, 1)
	for i := 0; i < m.r; i++ {
		out.data[i] = m.data[i*m.c+j]
	}
	return out, nil
}

func (m *Matrix) String() string {
	out := ""
	for i := 0; i < m.r; i++ {
		out += "["
		for j := 0; j < m.c; j++ {
			out = fmt.Sprintf("%s %f2", out, m.data[i*m.c+j])
		}
		out += " ]\n"
	}
	return out
}

//Equals compares the rt matrix element by element to this matrix testing for equality
func (m *Matrix) Equals(rt *Matrix) bool {
	if (rt == nil) ||
		(m.c != rt.c) ||
		(m.r != rt.r) {
		return false
	}
	for i := 0; i < len(m.data); i++ {
		if m.data[i] != rt.data[i] {
			return false
		}
	}
	return true
}

//Set assigns val such that matrix[r,c] = val. Assumes
//0 indexed rows and columns
func (m *Matrix) Set(r, c int, val float64) error {
	if (c >= m.c) || (c < 0) {
		return &InvalidColumnError{col: c}
	}
	if (r >= m.r) || (r < 0) {
		return &InvalidRowError{row: r}
	}

	fmt.Println("setting location", r, m.c, c, r*m.c+c)
	m.data[r*m.c+c] = val
	return nil
}

//Get retreives the scalar value at row r column c.
func (m *Matrix) Get(r, c int) (float64, error) {
	if (c >= m.c) || (c < 0) {
		return 0.0, &InvalidColumnError{col: c}
	}
	if (r >= m.r) || (r < 0) {
		return 0.0, &InvalidRowError{row: r}
	}
	return m.data[r*m.c+c], nil
}

//Clone makes a copy of a Matrix and returns it
func (m *Matrix) Clone() *Matrix {
	out := NewMatrix(m.r, m.c)
	copy(out.data, m.data)
	return out
}

type InvalidAddOperation struct{}

func (e *InvalidAddOperation) Error() string {
	return "row count and col count must match in both left and right"
}

//Add performs a matrix add with this matrix and rt and returns the result
//as a new matrix. This matrix is not modified.
func (m *Matrix) Add(rt *Matrix) (*Matrix, error) {
	if rt.c != m.c ||
		rt.r != m.r {
		return nil, &InvalidAddOperation{}
	}
	out := m.Clone()
	for i := 0; i < len(m.data); i++ {
		out.data[i] += rt.data[i]
	}
	return out, nil
}

//RowScalarMultiply multiplies row r by scalar v and returns the resulting matrix
func (m *Matrix) RowScalarMultiply(r int, v float64) (*Matrix, error) {
	if r < 0 || r >= m.r {
		return nil, &InvalidRowError{}
	}

	out := m.Clone()
	for j := 0; j < m.c; j++ {
		out.data[r*m.c+j] *= v
	}
	return out, nil
}

//RowInterchange swaps row r with row r1 in this matrix and returns the result
func (m *Matrix) RowInterchange(r, r1 int) (*Matrix, error) {
	if r < 0 || r >= m.r {
		return nil, &InvalidRowError{row: r}
	}
	if r1 < 0 || r1 >= m.r {
		return nil, &InvalidRowError{row: r1}
	}
	out := m.Clone()
	for j := 0; j < m.c; j++ {
		tmp := out.data[r*m.c+j]
		out.data[r*m.c+j] = out.data[r1*m.c+j]
		out.data[r1*m.c+j] = tmp
	}
	return out, nil
}

//RowMulAdd multiplies row r by scalar v and adds the result to row r1 and returns the result
func (m *Matrix) RowMulAdd(r int, v float64, r1 int) (*Matrix, error) {
	if r < 0 || r >= m.r {
		return nil, &InvalidRowError{row: r}
	}
	if r1 < 0 || r1 >= m.r {
		return nil, &InvalidRowError{row: r1}
	}
	out := m.Clone()
	for j := 0; j < m.c; j++ {
		out.data[r1*m.c+j] += (v * out.data[r*m.c+j])
	}
	return out, nil
}

type IncompatibleMatrixError struct{}

func (e *IncompatibleMatrixError) Error() string {
	return "Incompatible matrices for multiplication operation"
}

//Mul multiplies this matrix by the rt matrix and returns the result
func (m *Matrix) Mul(rt *Matrix) (*Matrix, error) {
	if (m.r != rt.c) || (m.c != rt.r) {
		return nil, &IncompatibleMatrixError{}
	}
	out := NewMatrix(m.r, rt.c)
	outRowOffset := 0
	mRowOffset := 0
	for i := 0; i < m.r; i++ {
		for j := 0; j < rt.c; j++ {
			rtRowOffset := 0
			for k := 0; k < m.c; k++ {
				out.data[outRowOffset+j] += (m.data[mRowOffset+k] * rt.data[rtRowOffset+j])
				rtRowOffset += rt.c
			}
		}
		outRowOffset += rt.c
		mRowOffset += m.c
	}
	return out, nil
}

//Transpose returns the matrix transpose of this matrix
func (m *Matrix) Transpose() *Matrix {
	out := NewMatrix(m.c, m.r)
	for i := 0; i < m.r; i++ {
		for j := 0; j < m.c; j++ {
			out.data[j*out.c+i] = m.data[i*m.c+j]
		}
	}
	return out
}

func (m *Matrix) AppendRows(rt *Matrix) (*Matrix, error) {
	if m.c != rt.c {
		return nil, &IncompatibleMatrixError{}
	}
	return nil, nil
}

func (m *Matrix) AppendColumns(rt *Matrix) (*Matrix, error) {
	if m.r != rt.r {
		return nil, &IncompatibleMatrixError{}
	}
	return nil, nil
}

//
// minimize cT*x
// subject to Ax = b
// x >= 0
//
type Tableau struct {
	//Cost vector
	C *Matrix
	//constraints
	A *Matrix
	//contraints
	B *Matrix
}
