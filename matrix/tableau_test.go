package matrix

import (
	"testing"
)

func TestTableauCreation(t *testing.T) {
	a := &Matrix{r: 2, c: 6, data: []float64{4, 2, -1, 0, 1, 0, 1, 4, 0, -1, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 6, data: []float64{0, 0, 0, 0, 1, 1}}
	tab, err := NewTableau(ct, a, b)
	if err != nil {
		t.Error(err)
		return
	}
	if tab.data.r != 3 && tab.data.c != 7 {
		t.Errorf("incorrect tableau dimensions, should be (3,7), got (%d,%d)", tab.data.r, tab.data.c)
	}
}

func TestCT(t *testing.T) {
	a := &Matrix{r: 2, c: 6, data: []float64{4, 2, -1, 0, 1, 0, 1, 4, 0, -1, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 6, data: []float64{0, 0, 0, 0, 1, 1}}
	tab, _ := NewTableau(ct, a, b)
	tct := tab.CT()
	for i := 0; i < len(ct.data); i++ {
		if tct.data[i] != ct.data[i] {
			t.Errorf("incorrect value, expected %.5f, got %.5f", ct.data[i], tct.data[i])
		}
	}
}

func TestA(t *testing.T) {
	a := &Matrix{r: 2, c: 6, data: []float64{4, 2, -1, 0, 1, 0, 1, 4, 0, -1, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 6, data: []float64{0, 0, 0, 0, 1, 1}}
	tab, _ := NewTableau(ct, a, b)
	ta := tab.A()
	for i := 0; i < len(a.data); i++ {
		if ta.data[i] != a.data[i] {
			t.Errorf("incorrect value, expected %.5f, got %.5f", a.data[i], ta.data[i])
		}
	}
}

func TestB(t *testing.T) {
	a := &Matrix{r: 2, c: 6, data: []float64{4, 2, -1, 0, 1, 0, 1, 4, 0, -1, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 6, data: []float64{0, 0, 0, 0, 1, 1}}
	tab, _ := NewTableau(ct, a, b)
	tb := tab.B()
	for i := 0; i < len(b.data); i++ {
		if tb.data[i] != b.data[i] {
			t.Errorf("incorrect value, expected %.5f, got %.5f", b.data[i], tb.data[i])
		}
	}
}

func TestPhase1(t *testing.T) {
	a := &Matrix{r: 2, c: 4, data: []float64{4, 2, -1, 0, 1, 4, 0, -1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 4, data: []float64{0, 0, 0, 0}}
	tb, err := phase1(ct, a, b)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{1, 0, -(float64(2) / 7.0), float64(1) / 7.0, float64(2) / 7.0, -(float64(1) / 7.0), float64(18) / 7.0,
		0, 1, float64(1) / 14, -float64(2) / 7, -float64(1) / 14, float64(2) / 7, float64(6) / 7,
		0, 0, 0, 0, 1, 1, 0,
	}
	for i := 0; i < len(output); i++ {
		if tb.data.data[i] < (output[i]-precision) ||
			tb.data.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.5f, got %.5f", output[i], tb.data.data[i])
		}
	}
}

func TestPhase2(t *testing.T) {
	a := &Matrix{r: 2, c: 4, data: []float64{4, 2, -1, 0, 1, 4, 0, -1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 4, data: []float64{0, 0, 0, 0}}
	tb, err := phase1(ct, a, b)
	if err != nil {
		t.Error(err)
		return
	}
	ct2 := &Matrix{r: 1, c: 4, data: []float64{2, 3, 0, 0}}
	ans, err := phase2(tb, ct2)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{float64(18) / 7.0, float64(6) / 7.0}
	for i := 0; i < len(output); i++ {
		if ans.data[i] < (output[i]-precision) ||
			ans.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.5f, got %.5f", output[i], ans.data[i])
		}
	}
}

func TestLPSolve(t *testing.T) {
	a := &Matrix{r: 2, c: 4, data: []float64{4, 2, -1, 0, 1, 4, 0, -1}}
	b := &Matrix{r: 2, c: 1, data: []float64{12, 6}}
	ct := &Matrix{r: 1, c: 4, data: []float64{2, 3, 0, 0}}
	ans, err := LPSolve(ct, a, b)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{float64(18) / 7.0, float64(6) / 7.0}
	for i := 0; i < len(output); i++ {
		if ans.data[i] < (output[i]-precision) ||
			ans.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.5f, got %.5f", output[i], ans.data[i])
		}
	}
}

func TestLPSolve2(t *testing.T) {
	a := &Matrix{r: 2, c: 4, data: []float64{2, 1, 1, 0, 1, 4, 0, 1}}
	b := &Matrix{r: 2, c: 1, data: []float64{3, 4}}
	ct := &Matrix{r: 1, c: 4, data: []float64{-7, -6, 0, 0}}
	ans, err := LPSolve(ct, a, b)
	if err != nil {
		t.Error(err)
		return
	}
	output := []float64{float64(8) / 7.0, float64(5) / 7.0}
	for i := 0; i < len(output); i++ {
		if ans.data[i] < (output[i]-precision) ||
			ans.data[i] > (output[i]+precision) {
			t.Errorf("incorrect value, expected %.5f, got %.5f", output[i], ans.data[i])
		}
	}
}
