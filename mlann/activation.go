package mlann

import (
	"fmt"
	"math"
	"reflect"
	"strings"
)

func init() {
	registerActivation(activationRegistry, (*RELU)(nil))
	registerActivation(activationRegistry, (*RELUPrime)(nil))
	registerActivation(activationRegistry, (*Linear)(nil))
	registerActivation(activationRegistry, (*LinearPrime)(nil))
	registerActivation(activationRegistry, (*Sigmoid)(nil))
	registerActivation(activationRegistry, (*SigmoidPrime)(nil))
}

var activationRegistry = make(map[string]reflect.Type)

func registerActivation(registry map[string]reflect.Type, typedNil interface{}) {
	t := reflect.TypeOf(typedNil).Elem()
	name := fmt.Sprintf("%s", t)
	registry[strings.Trim(name, "mlann.")] = t
}

//NotDefinedError occurs when the requested feature is not defined in the system
type NotDefinedError struct{}

func (e *NotDefinedError) Error() string {
	return "requested Activation function not defined"
}

//ListActivations lists the names of the available features registered with the system
func ListActivations() []string {
	out := make([]string, 0)
	for k := range activationRegistry {
		out = append(out, k)
	}
	return out
}

func getType(name string, registry map[string]reflect.Type) (r reflect.Type, e error) {
	if t, ok := registry[name]; ok {
		r = t
		return
	}
	e = &NotDefinedError{}
	return
}

func getActivation(name string) (Activation, error) {
	t, err := getType(name, activationRegistry)
	if err != nil {
		return nil, err
	}
	v := reflect.New(t)
	f := v.Interface().(Activation)
	return f, nil
}

//Activation defines an activation function for a layer in ANN.
type Activation interface {
	Activate(input float64) float64
	Name() string
}

//RELU is a rectified linear unit where any input less than or equal to 0
// produces an output of 0.0 and a linear output otherwise
type RELU struct{}

//Activate applies the activation function to the input and returns the result
func (r *RELU) Activate(input float64) float64 {
	if input < 0 {
		return float64(0.01)
	}
	return input
}

//Name of the Activation function
func (r *RELU) Name() string {
	return "RELU"
}

//RELUPrime is the first derivative of the RELU activation function
type RELUPrime struct{}

//Activate computes the value of the derivative at input and returns the result
func (r *RELUPrime) Activate(input float64) float64 {
	if input <= 0.0 {
		return float64(0.0)
	}
	return 1.0
}

//Name of the Activation function
func (r *RELUPrime) Name() string {
	return "RELUPrime"
}

//Linear Activation function
type Linear struct{}

//Activate produces an output equal to it's input
func (l *Linear) Activate(input float64) float64 {
	return input
}

//Name of the Activation function
func (l *Linear) Name() string {
	return "Linear"
}

//LinearPrime produces the derivitive of the Linear activation function
//so 1.0 regardless of the input
type LinearPrime struct{}

//Activate produces 1.0 as the output regardless of the input.
func (lp *LinearPrime) Activate(input float64) float64 {
	return float64(1.0)
}

//Name of the Activation function
func (lp *LinearPrime) Name() string {
	return "LinearPrime"
}

//Sigmoid produces the sigmoid function output
type Sigmoid struct{}

//Activate produces 1 / (1 + Exp(-input))
func (s *Sigmoid) Activate(input float64) float64 {
	return float64(1.0) / (1.0 + math.Exp(-input))
}

//Name of the Activation function
func (s *Sigmoid) Name() string {
	return "Sigmoid"
}

//SigmoidPrime produces the first derivitive output of the sigmoid function
type SigmoidPrime struct{}

//Activate produces Sigmoid(x) * ( 1 - Sigmoid(x))
func (sp *SigmoidPrime) Activate(input float64) float64 {
	s := float64(1.0) / (1.0 + math.Exp(-input))
	return s * (float64(1.0) - s)
}

//Name of the Activation function
func (sp *SigmoidPrime) Name() string {
	return "SigmoidPrime"
}
