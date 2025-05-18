package main

import (
	"fmt"
)

func main() {
	// Get user input
	fmt.Printf("Please give your name: ")
	var name string
	fmt.Scanln(&name)
	fmt.Printf("Your name is: %s!\n", name)
}
