package main

import "fmt"

func main() {
	// Traditional for loop
	for i := 0; i < 10; i++ {
		fmt.Println(i*i, " ")
	}
	fmt.Println()

	i := 0

	for ok := true; ok; ok = (i != 10) {
		fmt.Println(i*i, " ")
		i++
	}
	fmt.Println()

	// For loop used as a while loop
	i = 0
	for {
		if i == 10 {
			break
		}
		fmt.Println(i*i, " ")
		i++
	}
	fmt.Println()

	// This is a slice but range also works with arrays
	aSlice := []int{-1, 2, 1, -1, 2, -2}
	for i, v := range aSlice {
		fmt.Println("Index:", i, "value: ", v)
	}
}
