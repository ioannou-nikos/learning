package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Please provide a command line argument.")
		return
	}
	argument := os.Args[1]
	switch argument {
	case "0":
		fmt.Println("Zero!")
	case "1":
		fmt.Println("One!")
	case "2", "3", "4":
		fmt.Println("Two, three or four!")
		fallthrough
	default:
		fmt.Println("Value:", argument)
	}
	value, err := strconv.Atoi(argument)
	if err != nil {
		fmt.Println("Cannot convert argument to integer.", argument)
		return
	}
	// No expression after switch
	switch {
	case value == 0:
		fmt.Println("Zero!")
	case value > 0:
		fmt.Println("Positive!")
	case value < 0:
		fmt.Println("Negative!")
	default:
		fmt.Println("This should not happen:", value)
	}
}
