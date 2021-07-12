package typeutil

import "llvm.org/llgo/third_party/gotools/go/types"

// Dependencies returns all dependencies of the specified packages.
//
// Dependent packages appear in topological order: if package P imports
// package Q, Q appears earlier than P in the result.
// The algorithm follows import statements in the order they
// appear in the source code, so the result is a total order.
//
func Dependencies(pkgs ...*types.Package) []*types.Package {
	var result []*types.Package
	seen := make(map[*types.Package]bool)
	var visit func(pkgs []*types.Package)
	visit = func(pkgs []*types.Package) {
		for _, p := range pkgs {
			if !seen[p] {
				seen[p] = true
				visit(p.Imports())
				result = append(result, p)
			}
		}
	}
	visit(pkgs)
	return result
}
