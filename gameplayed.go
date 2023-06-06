package tictactoe

import "bigfunbrewing.com/tensor"

// GamePlayed records the sequence of Positions that occurred during the game along with
// the outcome.
type GamePlayed struct {
	positions []Position
	outcome   float64
}

func (gp *GamePlayed) Append(position Position) {
	gp.positions = append(gp.positions, position)
}

func (gp *GamePlayed) Positions() []Position {
	return gp.positions
}

func (gp *GamePlayed) Outcome() float64 {
	return gp.outcome
}

func NewGamePlayed() *GamePlayed {
	return &GamePlayed{positions: make([]Position, 0)}
}

// ToSample converts the GamePlayed into a sample where the slice of reward
// has already decayed the outcome back to the first move of the game.
func (gp *GamePlayed) ToSample(reward []float64) *tensor.Sample[float64] {
	out := gp.positions[0]
	for i := 1; i < len(gp.positions); i++ {
		(*tensor.Tensor[float64])(out).Append(1, gp.positions[i])
	}
	sample := tensor.NewSample[float64](out, tensor.New(tensor.WithShape[float64](1, len(reward)), tensor.WithBacking(reward)))
	return sample
}

// ToSequenceSample converts a GamePlayed into a SequenceSample suitable for an RNN
// reward corresponds to the outcome of the game played.
func (gp *GamePlayed) ToSequenceSample(reward float64) *tensor.SequenceSample {
	pos := make([]*tensor.Tensor[float64], len(gp.positions))
	for i := range gp.positions {
		pos[i] = (*tensor.Tensor[float64])(gp.positions[i])
	}
	return tensor.MakeSequenceSample(pos, reward)
}

// Rotate the GamePlayed so that all translationally equivalent Games are
// represented as well.
func (gp *GamePlayed) Rotate(inc int) *GamePlayed {
	out := NewGamePlayed()
	out.outcome = gp.outcome
	out.positions = make([]Position, len(gp.positions))
	for i := range gp.positions {
		out.positions[i] = rotate(gp.positions[i], inc)
	}

	return out
}

func rotate(in *tensor.Tensor[float64], inc int) *tensor.Tensor[float64] {
	out := in.Clone()
	switch inc % 4 {
	case 1: // 90 degrees
		// 0,3,6    2,1,0 //0,1,2     6,3,0
		// 1,4,7 -> 5,4,3 //3,4,5 ->  7,4,1
		// 2,5,8    8,7,6 //6,7,8     8,5,2
		out.Set(in.Get(2, 0), 0, 0)
		out.Set(in.Get(5, 0), 1, 0)
		out.Set(in.Get(8, 0), 2, 0)
		out.Set(in.Get(1, 0), 3, 0)
		out.Set(in.Get(4, 0), 4, 0)
		out.Set(in.Get(7, 0), 5, 0)
		out.Set(in.Get(0, 0), 6, 0)
		out.Set(in.Get(3, 0), 7, 0)
		out.Set(in.Get(6, 0), 8, 0)

		out.Set(in.Get(2+9, 0), 9, 0)
		out.Set(in.Get(5+9, 0), 10, 0)
		out.Set(in.Get(8+9, 0), 11, 0)
		out.Set(in.Get(1+9, 0), 12, 0)
		out.Set(in.Get(4+9, 0), 13, 0)
		out.Set(in.Get(7+9, 0), 14, 0)
		out.Set(in.Get(0+9, 0), 15, 0)
		out.Set(in.Get(3+9, 0), 16, 0)
		out.Set(in.Get(6+9, 0), 17, 0)
	case 2: // 180 degrees
		// 0,3,6    8,5,2
		// 1,4,7 -> 7,4,1
		// 2,5,8    6,3,0
		out.Set(in.Get(8, 0), 0, 0)
		out.Set(in.Get(7, 0), 1, 0)
		out.Set(in.Get(6, 0), 2, 0)
		out.Set(in.Get(5, 0), 3, 0)
		out.Set(in.Get(4, 0), 4, 0)
		out.Set(in.Get(3, 0), 5, 0)
		out.Set(in.Get(2, 0), 6, 0)
		out.Set(in.Get(1, 0), 7, 0)
		out.Set(in.Get(0, 0), 8, 0)

		out.Set(in.Get(8+9, 0), 9, 0)
		out.Set(in.Get(7+9, 0), 10, 0)
		out.Set(in.Get(6+9, 0), 11, 0)
		out.Set(in.Get(5+9, 0), 12, 0)
		out.Set(in.Get(4+9, 0), 13, 0)
		out.Set(in.Get(3+9, 0), 14, 0)
		out.Set(in.Get(2+9, 0), 15, 0)
		out.Set(in.Get(1+9, 0), 16, 0)
		out.Set(in.Get(0+9, 0), 17, 0)
	case 3: // 270 degrees
		// 0,3,6    6,7,8
		// 1,4,7 -> 3,4,5
		// 2,5,8    0,1,2
		out.Set(in.Get(6, 0), 0, 0)
		out.Set(in.Get(3, 0), 1, 0)
		out.Set(in.Get(0, 0), 2, 0)
		out.Set(in.Get(7, 0), 3, 0)
		out.Set(in.Get(4, 0), 4, 0)
		out.Set(in.Get(1, 0), 5, 0)
		out.Set(in.Get(8, 0), 6, 0)
		out.Set(in.Get(5, 0), 7, 0)
		out.Set(in.Get(2, 0), 8, 0)

		out.Set(in.Get(8+9, 0), 9, 0)
		out.Set(in.Get(7+9, 0), 10, 0)
		out.Set(in.Get(6+9, 0), 11, 0)
		out.Set(in.Get(5+9, 0), 12, 0)
		out.Set(in.Get(4+9, 0), 13, 0)
		out.Set(in.Get(3+9, 0), 14, 0)
		out.Set(in.Get(2+9, 0), 15, 0)
		out.Set(in.Get(1+9, 0), 16, 0)
		out.Set(in.Get(0+9, 0), 17, 0)
	}
	return out
}
