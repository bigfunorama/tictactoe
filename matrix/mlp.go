package matrix

type Activation interface {
	Activate(input float64) float64
}

type RELU struct{}

func (r *RELU) Activate(input float64) float64 {
	if input < 0 {
		return float64(0.0)
	}
	return input
}

type Layer struct {
	h Activation
	w *Matrix
	b *Matrix
}

//Activate returns l.h(l.w * x + l.b)
func (l *Layer) Activate(x *Matrix) (*Matrix, error) {
	tmp, err := l.w.Mul(x)
	if err != nil {
		return nil, err
	}
	tmp, err = tmp.Add(l.b)
	if err != nil {
		return nil, err
	}
	return tmp.Apply(l.h), nil
}
