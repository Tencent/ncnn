// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"fmt"
	"os"

	"llvm.org/llgo/third_party/gotools/go/loader"
	"llvm.org/llgo/third_party/gotools/go/ssa"
)

// This program demonstrates how to run the SSA builder on a "Hello,
// World!" program and shows the printed representation of packages,
// functions and instructions.
//
// Within the function listing, the name of each BasicBlock such as
// ".0.entry" is printed left-aligned, followed by the block's
// Instructions.
//
// For each instruction that defines an SSA virtual register
// (i.e. implements Value), the type of that value is shown in the
// right column.
//
// Build and run the ssadump.go program if you want a standalone tool
// with similar functionality. It is located at
// golang.org/x/tools/cmd/ssadump.
//
func Example() {
	const hello = `
package main

import "fmt"

const message = "Hello, World!"

func main() {
	fmt.Println(message)
}
`
	var conf loader.Config

	// Parse the input file.
	file, err := conf.ParseFile("hello.go", hello)
	if err != nil {
		fmt.Print(err) // parse error
		return
	}

	// Create single-file main package.
	conf.CreateFromFiles("main", file)

	// Load the main package and its dependencies.
	iprog, err := conf.Load()
	if err != nil {
		fmt.Print(err) // type error in some package
		return
	}

	// Create SSA-form program representation.
	prog := ssa.Create(iprog, ssa.SanityCheckFunctions)
	mainPkg := prog.Package(iprog.Created[0].Pkg)

	// Print out the package.
	mainPkg.WriteTo(os.Stdout)

	// Build SSA code for bodies of functions in mainPkg.
	mainPkg.Build()

	// Print out the package-level functions.
	mainPkg.Func("init").WriteTo(os.Stdout)
	mainPkg.Func("main").WriteTo(os.Stdout)

	// Output:
	//
	// package main:
	//   func  init       func()
	//   var   init$guard bool
	//   func  main       func()
	//   const message    message = "Hello, World!":untyped string
	//
	// # Name: main.init
	// # Package: main
	// # Synthetic: package initializer
	// func init():
	// 0:                                                                entry P:0 S:2
	// 	t0 = *init$guard                                                   bool
	// 	if t0 goto 2 else 1
	// 1:                                                           init.start P:1 S:1
	// 	*init$guard = true:bool
	// 	t1 = fmt.init()                                                      ()
	// 	jump 2
	// 2:                                                            init.done P:2 S:0
	// 	return
	//
	// # Name: main.main
	// # Package: main
	// # Location: hello.go:8:6
	// func main():
	// 0:                                                                entry P:0 S:0
	// 	t0 = new [1]interface{} (varargs)                       *[1]interface{}
	// 	t1 = &t0[0:int]                                            *interface{}
	// 	t2 = make interface{} <- string ("Hello, World!":string)    interface{}
	// 	*t1 = t2
	// 	t3 = slice t0[:]                                          []interface{}
	// 	t4 = fmt.Println(t3...)                              (n int, err error)
	// 	return
}
