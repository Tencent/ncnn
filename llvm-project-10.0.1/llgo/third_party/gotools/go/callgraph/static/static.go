// Package static computes the call graph of a Go program containing
// only static call edges.
package static // import "llvm.org/llgo/third_party/gotools/go/callgraph/static"

import (
	"llvm.org/llgo/third_party/gotools/go/callgraph"
	"llvm.org/llgo/third_party/gotools/go/ssa"
	"llvm.org/llgo/third_party/gotools/go/ssa/ssautil"
)

// CallGraph computes the call graph of the specified program
// considering only static calls.
//
func CallGraph(prog *ssa.Program) *callgraph.Graph {
	cg := callgraph.New(nil) // TODO(adonovan) eliminate concept of rooted callgraph

	// TODO(adonovan): opt: use only a single pass over the ssa.Program.
	for f := range ssautil.AllFunctions(prog) {
		fnode := cg.CreateNode(f)
		for _, b := range f.Blocks {
			for _, instr := range b.Instrs {
				if site, ok := instr.(ssa.CallInstruction); ok {
					if g := site.Common().StaticCallee(); g != nil {
						gnode := cg.CreateNode(g)
						callgraph.AddEdge(fnode, site, gnode)
					}
				}
			}
		}
	}

	return cg
}
