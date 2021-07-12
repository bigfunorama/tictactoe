package mlann

import "testing"

func TestListActivations(t *testing.T) {
	acts := ListActivations()
	if len(acts) == 0 {
		t.Error("no activations found")
	}
}

func TestGetActivation(t *testing.T) {
	act, err := getActivation("Linear")
	if err != nil {
		t.Error(err.Error())
	}
	if act.Name() != "Linear" {
		t.Errorf("expected %s, got %s", "Linear", act.Name())
	}
}
