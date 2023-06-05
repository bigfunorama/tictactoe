module bigfunbrewing.com/tictactoe

go 1.19

replace bigfunbrewing.com/mlann => ../mlann

replace bigfunbrewing.com/tensor => ../tensor

require (
	bigfunbrewing.com/mlann v0.0.0-20220424021550-ca53ccf8484f
	bigfunbrewing.com/tensor v0.0.0-00010101000000-000000000000
)

require gonum.org/v1/gonum v0.12.0 // indirect
