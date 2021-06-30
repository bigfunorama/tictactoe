package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

func readFile(fname string) (string, error) {
	f, err := os.Open("./plot.html")
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

func dataHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/fetchdata" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	if r.Method != "GET" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	f, err := os.Open("./test.csv")
	if err != nil {
		fmt.Fprintf(w, "Error loading html")
	}
	defer f.Close()
	in := bufio.NewReader(f)
	b, err := ioutil.ReadAll(in)
	if err != nil {
		fmt.Fprintf(w, "Error reading html")
	}
	fmt.Fprintf(w, string(b))
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
