package matrix

import "fmt"

const precision = 1e-9

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
			out = fmt.Sprintf("%s %.5f", out, m.data[i*m.c+j])
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
		if m.data[i] <= (rt.data[i]-precision) ||
			m.data[i] >= (rt.data[i]+precision) {
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

type SingularMatrixError struct{}

func (e *SingularMatrixError) Error() string {
	return "Input Matrix was singular, inverse does not exist"
}

//Inverse finds the inverse of this matrix and returns it.
//if the matrix is singular, Inverse returns a singular matrix error.
func (m *Matrix) Inverse() (*Matrix, error) {
	if m.c != m.r {
		return nil, fmt.Errorf("Must be a square matrix to have an inverse")
	}
	augmented, err := m.AppendColumns(Identity(m.r))
	if err != nil {
		return nil, err
	}
	fmt.Println("start")
	fmt.Println(augmented)
	for i := 0; i < augmented.r; i++ {
		augmented, err = augmented.Pivot(i, i)
		if err != nil {
			return nil, err
		}
	}
	fmt.Println(augmented)
	return augmented.ExtractMatrix(0, m.c, m.r, augmented.c)
}

//AppendRows combines rows of the rt matrix after the last of m and return it
func (m *Matrix) AppendRows(rt *Matrix) (*Matrix, error) {
	if m.c != rt.c {
		return nil, &IncompatibleMatrixError{}
	}

	out := NewMatrix(m.r+rt.r, m.c)
	offset := 0
	mOffset := 0
	for j := 0; j < m.r; j++ {
		copy(out.data[offset:offset+m.c], m.data[mOffset:mOffset+m.c])
		offset += m.c
		mOffset += m.c
	}
	rtOffset := 0
	for j := 0; j < rt.r; j++ {
		copy(out.data[offset:offset+m.c], rt.data[rtOffset:rtOffset+rt.c])
		offset += rt.c
		rtOffset += rt.c
	}
	return out, nil
}

//AppendColumns appends the columns of matrix rt to matrix m and returns the results.
func (m *Matrix) AppendColumns(rt *Matrix) (*Matrix, error) {
	if m.r != rt.r {
		return nil, &IncompatibleMatrixError{}
	}
	out := NewMatrix(m.r, (m.c + rt.c))

	mOffset := 0
	rtOffset := 0
	outOffset := 0
	for i := 0; i < m.r; i++ {
		copy(out.data[outOffset:(outOffset+m.c)], m.data[mOffset:(mOffset+m.c)])
		mOffset += m.c
		outOffset += m.c
		copy(out.data[outOffset:(outOffset+rt.c)], rt.data[rtOffset:(rtOffset+rt.c)])
		rtOffset += rt.c
		outOffset += rt.c
	}
	return out, nil
}

//ExtractMatrix extracts the sub-matrix from this matrix that starts at ulRow, ulCol
// and ends lrRow, lrCol where lrRow is the index of the next row after the sub-matrix
//ends and lrCol is the first column after the sub-matrix ends.
// Some requirements on the inputs.
// ulCol <= lrCol
// ulRow <= lrRow
// 0 <= ulCol, lrCol
func (m *Matrix) ExtractMatrix(ulRow, ulCol, lrRow, lrCol int) (*Matrix, error) {
	if ulRow > lrRow ||
		ulCol > lrCol ||
		ulRow < 0 || ulRow >= m.r ||
		ulCol < 0 || ulCol >= m.c ||
		lrRow < 0 || lrRow > m.r ||
		lrCol < 0 || lrCol > m.c {
		return nil, fmt.Errorf("invalid arguments")
	}
	rows := lrRow - ulRow
	cols := lrCol - ulCol
	out := NewMatrix(rows, cols)
	offset := 0
	mOffset := ulRow*m.c + ulCol
	for i := 0; i < rows; i++ {
		copy(out.data[offset:(offset+cols)], m.data[mOffset:(mOffset+cols)])
		offset += cols
		mOffset += m.c
	}
	return out, nil
}

//Pivot applies elementary row operations to zero out all entries in col c
//except the entry at row r column c. This entry must not be 0
func (m *Matrix) Pivot(r, c int) (*Matrix, error) {
	if r >= m.r || r < 0 ||
		c >= m.c || c < 0 ||
		(m.data[r*m.c+c] <= precision && m.data[r*m.c+c] >= -precision) {
		return nil, fmt.Errorf("invalid arguments")
	}
	augmented := m.Clone()
	fmt.Println("start")
	fmt.Println(augmented)

	//normalize row r relative to the value at r,c
	div := float64(1) / augmented.data[r*augmented.c+c]
	augmented, _ = augmented.RowScalarMultiply(r, div)
	fmt.Println("convert to 1.0", r, c)
	fmt.Println(augmented)
	jOffset := 0
	for j := 0; j < augmented.r; j++ {
		if r != j {
			arg := augmented.data[jOffset+c]
			augmented, _ = augmented.RowMulAdd(r, -arg, j)
			fmt.Println("zero out", j, c)
			fmt.Println(augmented)
		}
		jOffset += augmented.c
	}
	return augmented, nil
}
