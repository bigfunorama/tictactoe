package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

func readFile(fname string) (string, error) {
	f, err := os.Open(fname)
	if err != nil {
		return "", err
	}
	defer f.Close()

	in := bufio.NewReader(f)
	b, err := ioutil.ReadAll(in)
	if err != nil {
		return "", err
	}
	return string(b), err
}

func plotHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/plot" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	if r.Method != "GET" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	out, err := readFile("./plot.html")
	if err != nil {
		http.Error(w, "internal error, "+err.Error(), http.StatusInternalServerError)
	}
	fmt.Fprintf(w, out)
}

func checkLabels(xlabel, ylabel, zlabel string) error {
	out := "missing labels: "
	if xlabel == "" {
		out += "xlabel "
	}
	if ylabel == "" {
		out += "ylabel "
	}
	if zlabel == "" {
		out += "zlabel "
	}

	if out != "missing labels: " {
		return fmt.Errorf(out)
	}
	return nil
}

func readCSV(fname, xlabel, ylabel, zlabel string) (string, error) {
	log.Println("loading input, ", fname)
	f, err := os.Open(fname)
	if err != nil {
		return "", err
	}
	in := bufio.NewReader(f)
	cc := csv.NewReader(in)
	keys, err := cc.Read()
	err = checkLabels(xlabel, ylabel, zlabel)
	if err != nil {
		return "", err
	}
	cols := []int{-1, -1, -1}
	for idx := 0; idx < len(keys); idx++ {
		if keys[idx] == xlabel {
			cols[0] = idx
		}
		if keys[idx] == ylabel {
			cols[1] = idx
		}
		if keys[idx] == zlabel {
			cols[2] = idx
		}
	}

	out := xlabel + "," + ylabel + "," + zlabel + "\n"
	row, err := cc.Read()
	for err == nil {
		rr := row[cols[0]] + "," + row[cols[1]] + "," + row[cols[2]] + "\n"
		out += rr
		row, err = cc.Read()
	}
	return out, nil
}

func dataHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/fetchdata" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	if r.Method != "GET" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	vals := r.URL.Query()
	xlabel := vals.Get("x")
	ylabel := vals.Get("y")
	zlabel := vals.Get("z")

	var out string
	var err error
	e := checkLabels(xlabel, ylabel, zlabel)
	if e != nil {
		out, err = readFile("./test.csv")
	} else {
		out, err = readCSV("./test.csv", xlabel, ylabel, zlabel)
	}
	if err != nil {
		http.Error(w, "internal error, "+err.Error(), http.StatusInternalServerError)
	}
	fmt.Fprintf(w, out)
}

func trainingHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/trainingData" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	if r.Method != "GET" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	out, err := readFile("./training.json")
	if err != nil {
		http.Error(w, "internal error, "+err.Error(), http.StatusInternalServerError)
	}
	fmt.Fprintf(w, out)
}

func main() {
	http.HandleFunc("/plot", plotHandler)
	http.HandleFunc("/fetchdata", dataHandler)
	http.HandleFunc("/trainingData", trainingHandler)
	fmt.Printf("Starting server at port 8080\n")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
