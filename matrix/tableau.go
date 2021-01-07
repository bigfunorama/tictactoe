package matrix

import "fmt"

// Tableau matrix with special rows and columns corresponding to an optimization problem
// minimize cT*x
// subject to Ax = b
// x >= 0
//
type Tableau struct {
	data *Matrix
}

//NewTableau constructs a canonical Tableau from the matrices ct, a and b.
//if the provided matrices are incompatible with this operation, the function
//will return an error.
func NewTableau(ct, a, b *Matrix) (*Tableau, error) {
	t, err := a.AppendColumns(b)
	if err != nil {
		return nil, err
	}
	cost, err := ct.AppendColumns(&Matrix{r: 1, c: 1, data: []float64{0}})
	if err != nil {
		return nil, err
	}
	data, _ := t.AppendRows(cost)
	fmt.Println(data)
	return &Tableau{data: data}, nil
}

//CT extracts c transpose from the tableau
func (t *Tableau) CT() *Matrix {
	ct, _ := t.data.ExtractMatrix(t.data.r-1, 0, t.data.r, t.data.c-1)
	return ct
}

//A extracts the matrix A from the tableau
func (t *Tableau) A() *Matrix {
	a, _ := t.data.ExtractMatrix(0, 0, t.data.r-1, t.data.c-1)
	return a
}

//B extract the matrix b from the tableau
func (t *Tableau) B() *Matrix {
	b, _ := t.data.ExtractMatrix(0, t.data.c-1, t.data.r-1, t.data.c)
	return b
}

func (t *Tableau) maxNegativeRCC() (int, error) {
	ct := t.CT()
	col := 0
	val := float64(0)
	for i := 0; i < ct.c; i++ {
		if ct.data[i] < val {
			col = i
			val = ct.data[i]
		}
	}
	if val >= 0 {
		return col, fmt.Errorf("no negative elements")
	}
	return col, nil
}

//pivot around column col
func (t *Tableau) pivot(col int) error {
	tmp, err := t.data.Pivot(col, col)
	if err != nil {
		return err
	}
	t.data = tmp
	return nil
}

//phase1 executes the first phase of the two phase simplex algorithm
//returning the resulting Tableau. This process constructs the canonical
//Tableau and converts to a basic feasible solution to the original LP problem.
func phase1(ct, a, b *Matrix) (*Tableau, error) {
	i := Identity(a.r)
	tmp, _ := a.AppendColumns(i)
	top, err := tmp.AppendColumns(b)
	if err != nil {
		return nil, err
	}

	ones := NewMatrix(1, a.r+1)
	for idx := 0; idx < a.r; idx++ {
		ones.data[idx] = 1.0
	}
	ones.data[a.r] = 0

	cprime, _ := ct.AppendColumns(ones)
	data, err := top.AppendRows(cprime)
	if err != nil {
		return nil, err
	}

	for idx := 0; idx < a.r; idx++ {
		data, err = data.Pivot(idx, idx+a.c)
		if err != nil {
			return nil, err
		}
	}

	t := &Tableau{data: data}
	return optimize(t)
}

func optimize(t *Tableau) (*Tableau, error) {
	col, err := t.maxNegativeRCC()
	for err == nil {
		err = t.pivot(col)
		if err == nil {
			col, err = t.maxNegativeRCC()
		}
	}
	return t, nil
}

//phase2 executes the second phase of the two phase simplex algorithm.
//The tableau is expected to contain a basic feasible solution. Phase 2
//pivots from the initial basic feasible solution to the optimal basic
//feasible solution.
func phase2(tb *Tableau, ct *Matrix) (*Matrix, error) {
	fmt.Println("phase 2")
	ap1 := tb.A()
	ap1, _ = ap1.ExtractMatrix(0, 0, 2, 4)
	bp1 := tb.B()
	data, err := ap1.AppendColumns(bp1)
	if err != nil {
		return nil, err
	}
	fmt.Println(data)
	c, _ := ct.AppendColumns(NewMatrix(1, 1))
	data, err = data.AppendRows(c)
	if err != nil {
		return nil, err
	}
	fmt.Println(data)
	t2, err := optimize(&Tableau{data: data})
	if err != nil {
		return nil, err
	}
	return t2.B(), nil
}

//LPSolve produces the optimal basic feasible solution to the given standard
//form LP problem.
func LPSolve(ct, a, b *Matrix) (*Matrix, error) {
	pct := NewMatrix(1, a.c)
	t, err := phase1(pct, a, b)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}
	return phase2(t, ct)
}
