// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"strings"
	"testing"
)

var validFiles = []string{
	"parser.go",
	"parser_test.go",
	"error_test.go",
	"short_test.go",
}

func TestParse(t *testing.T) {
	for _, filename := range validFiles {
		_, err := ParseFile(token.NewFileSet(), filename, nil, DeclarationErrors)
		if err != nil {
			t.Fatalf("ParseFile(%s): %v", filename, err)
		}
	}
}

func nameFilter(filename string) bool {
	switch filename {
	case "parser.go", "interface.go", "parser_test.go":
		return true
	case "parser.go.orig":
		return true // permit but should be ignored by ParseDir
	}
	return false
}

func dirFilter(f os.FileInfo) bool { return nameFilter(f.Name()) }

func TestParseDir(t *testing.T) {
	path := "."
	pkgs, err := ParseDir(token.NewFileSet(), path, dirFilter, 0)
	if err != nil {
		t.Fatalf("ParseDir(%s): %v", path, err)
	}
	if n := len(pkgs); n != 1 {
		t.Errorf("got %d packages; want 1", n)
	}
	pkg := pkgs["parser"]
	if pkg == nil {
		t.Errorf(`package "parser" not found`)
		return
	}
	if n := len(pkg.Files); n != 3 {
		t.Errorf("got %d package files; want 3", n)
	}
	for filename := range pkg.Files {
		if !nameFilter(filename) {
			t.Errorf("unexpected package file: %s", filename)
		}
	}
}

func TestParseExpr(t *testing.T) {
	// just kicking the tires:
	// a valid arithmetic expression
	src := "a + b"
	x, err := ParseExpr(src)
	if err != nil {
		t.Errorf("ParseExpr(%q): %v", src, err)
	}
	// sanity check
	if _, ok := x.(*ast.BinaryExpr); !ok {
		t.Errorf("ParseExpr(%q): got %T, want *ast.BinaryExpr", src, x)
	}

	// a valid type expression
	src = "struct{x *int}"
	x, err = ParseExpr(src)
	if err != nil {
		t.Errorf("ParseExpr(%q): %v", src, err)
	}
	// sanity check
	if _, ok := x.(*ast.StructType); !ok {
		t.Errorf("ParseExpr(%q): got %T, want *ast.StructType", src, x)
	}

	// an invalid expression
	src = "a + *"
	if _, err := ParseExpr(src); err == nil {
		t.Errorf("ParseExpr(%q): got no error", src)
	}

	// a valid expression followed by extra tokens is invalid
	src = "a[i] := x"
	if _, err := ParseExpr(src); err == nil {
		t.Errorf("ParseExpr(%q): got no error", src)
	}

	// a semicolon is not permitted unless automatically inserted
	src = "a + b\n"
	if _, err := ParseExpr(src); err != nil {
		t.Errorf("ParseExpr(%q): got error %s", src, err)
	}
	src = "a + b;"
	if _, err := ParseExpr(src); err == nil {
		t.Errorf("ParseExpr(%q): got no error", src)
	}

	// various other stuff following a valid expression
	const validExpr = "a + b"
	const anything = "dh3*#D)#_"
	for _, c := range "!)]};," {
		src := validExpr + string(c) + anything
		if _, err := ParseExpr(src); err == nil {
			t.Errorf("ParseExpr(%q): got no error", src)
		}
	}

	// ParseExpr must not crash
	for _, src := range valids {
		ParseExpr(src)
	}
}

func TestColonEqualsScope(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `package p; func f() { x, y, z := x, y, z }`, 0)
	if err != nil {
		t.Fatal(err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.AssignStmt)
	for _, v := range as.Rhs {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, v := range as.Lhs {
		id := v.(*ast.Ident)
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

func TestVarScope(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `package p; func f() { var x, y, z = x, y, z }`, 0)
	if err != nil {
		t.Fatal(err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.DeclStmt).Decl.(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
	for _, v := range as.Values {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, id := range as.Names {
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

func TestObjects(t *testing.T) {
	const src = `
package p
import fmt "fmt"
const pi = 3.14
type T struct{}
var x int
func f() { L: }
`

	f, err := ParseFile(token.NewFileSet(), "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	objects := map[string]ast.ObjKind{
		"p":   ast.Bad, // not in a scope
		"fmt": ast.Bad, // not resolved yet
		"pi":  ast.Con,
		"T":   ast.Typ,
		"x":   ast.Var,
		"int": ast.Bad, // not resolved yet
		"f":   ast.Fun,
		"L":   ast.Lbl,
	}

	ast.Inspect(f, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok {
			obj := ident.Obj
			if obj == nil {
				if objects[ident.Name] != ast.Bad {
					t.Errorf("no object for %s", ident.Name)
				}
				return true
			}
			if obj.Name != ident.Name {
				t.Errorf("names don't match: obj.Name = %s, ident.Name = %s", obj.Name, ident.Name)
			}
			kind := objects[ident.Name]
			if obj.Kind != kind {
				t.Errorf("%s: obj.Kind = %s; want %s", ident.Name, obj.Kind, kind)
			}
		}
		return true
	})
}

func TestUnresolved(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `
package p
//
func f1a(int)
func f2a(byte, int, float)
func f3a(a, b int, c float)
func f4a(...complex)
func f5a(a s1a, b ...complex)
//
func f1b(*int)
func f2b([]byte, (int), *float)
func f3b(a, b *int, c []float)
func f4b(...*complex)
func f5b(a s1a, b ...[]complex)
//
type s1a struct { int }
type s2a struct { byte; int; s1a }
type s3a struct { a, b int; c float }
//
type s1b struct { *int }
type s2b struct { byte; int; *float }
type s3b struct { a, b *s3b; c []float }
`, 0)
	if err != nil {
		t.Fatal(err)
	}

	want := "int " + // f1a
		"byte int float " + // f2a
		"int float " + // f3a
		"complex " + // f4a
		"complex " + // f5a
		//
		"int " + // f1b
		"byte int float " + // f2b
		"int float " + // f3b
		"complex " + // f4b
		"complex " + // f5b
		//
		"int " + // s1a
		"byte int " + // s2a
		"int float " + // s3a
		//
		"int " + // s1a
		"byte int float " + // s2a
		"float " // s3a

	// collect unresolved identifiers
	var buf bytes.Buffer
	for _, u := range f.Unresolved {
		buf.WriteString(u.Name)
		buf.WriteByte(' ')
	}
	got := buf.String()

	if got != want {
		t.Errorf("\ngot:  %s\nwant: %s", got, want)
	}
}

var imports = map[string]bool{
	`"a"`:        true,
	"`a`":        true,
	`"a/b"`:      true,
	`"a.b"`:      true,
	`"m\x61th"`:  true,
	`"greek/αβ"`: true,
	`""`:         false,

	// Each of these pairs tests both `` vs "" strings
	// and also use of invalid characters spelled out as
	// escape sequences and written directly.
	// For example `"\x00"` tests import "\x00"
	// while "`\x00`" tests import `<actual-NUL-byte>`.
	`"\x00"`:     false,
	"`\x00`":     false,
	`"\x7f"`:     false,
	"`\x7f`":     false,
	`"a!"`:       false,
	"`a!`":       false,
	`"a b"`:      false,
	"`a b`":      false,
	`"a\\b"`:     false,
	"`a\\b`":     false,
	"\"`a`\"":    false,
	"`\"a\"`":    false,
	`"\x80\x80"`: false,
	"`\x80\x80`": false,
	`"\xFFFD"`:   false,
	"`\xFFFD`":   false,
}

func TestImports(t *testing.T) {
	for path, isValid := range imports {
		src := fmt.Sprintf("package p; import %s", path)
		_, err := ParseFile(token.NewFileSet(), "", src, 0)
		switch {
		case err != nil && isValid:
			t.Errorf("ParseFile(%s): got %v; expected no error", src, err)
		case err == nil && !isValid:
			t.Errorf("ParseFile(%s): got no error; expected one", src)
		}
	}
}

func TestCommentGroups(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `
package p /* 1a */ /* 1b */      /* 1c */ // 1d
/* 2a
*/
// 2b
const pi = 3.1415
/* 3a */ // 3b
/* 3c */ const e = 2.7182

// Example from issue 3139
func ExampleCount() {
	fmt.Println(strings.Count("cheese", "e"))
	fmt.Println(strings.Count("five", "")) // before & after each rune
	// Output:
	// 3
	// 5
}
`, ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	expected := [][]string{
		{"/* 1a */", "/* 1b */", "/* 1c */", "// 1d"},
		{"/* 2a\n*/", "// 2b"},
		{"/* 3a */", "// 3b", "/* 3c */"},
		{"// Example from issue 3139"},
		{"// before & after each rune"},
		{"// Output:", "// 3", "// 5"},
	}
	if len(f.Comments) != len(expected) {
		t.Fatalf("got %d comment groups; expected %d", len(f.Comments), len(expected))
	}
	for i, exp := range expected {
		got := f.Comments[i].List
		if len(got) != len(exp) {
			t.Errorf("got %d comments in group %d; expected %d", len(got), i, len(exp))
			continue
		}
		for j, exp := range exp {
			got := got[j].Text
			if got != exp {
				t.Errorf("got %q in group %d; expected %q", got, i, exp)
			}
		}
	}
}

func getField(file *ast.File, fieldname string) *ast.Field {
	parts := strings.Split(fieldname, ".")
	for _, d := range file.Decls {
		if d, ok := d.(*ast.GenDecl); ok && d.Tok == token.TYPE {
			for _, s := range d.Specs {
				if s, ok := s.(*ast.TypeSpec); ok && s.Name.Name == parts[0] {
					if s, ok := s.Type.(*ast.StructType); ok {
						for _, f := range s.Fields.List {
							for _, name := range f.Names {
								if name.Name == parts[1] {
									return f
								}
							}
						}
					}
				}
			}
		}
	}
	return nil
}

// Don't use ast.CommentGroup.Text() - we want to see exact comment text.
func commentText(c *ast.CommentGroup) string {
	var buf bytes.Buffer
	if c != nil {
		for _, c := range c.List {
			buf.WriteString(c.Text)
		}
	}
	return buf.String()
}

func checkFieldComments(t *testing.T, file *ast.File, fieldname, lead, line string) {
	f := getField(file, fieldname)
	if f == nil {
		t.Fatalf("field not found: %s", fieldname)
	}
	if got := commentText(f.Doc); got != lead {
		t.Errorf("got lead comment %q; expected %q", got, lead)
	}
	if got := commentText(f.Comment); got != line {
		t.Errorf("got line comment %q; expected %q", got, line)
	}
}

func TestLeadAndLineComments(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `
package p
type T struct {
	/* F1 lead comment */
	//
	F1 int  /* F1 */ // line comment
	// F2 lead
	// comment
	F2 int  // F2 line comment
	// f3 lead comment
	f3 int  // f3 line comment
}
`, ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	checkFieldComments(t, f, "T.F1", "/* F1 lead comment *///", "/* F1 */// line comment")
	checkFieldComments(t, f, "T.F2", "// F2 lead// comment", "// F2 line comment")
	checkFieldComments(t, f, "T.f3", "// f3 lead comment", "// f3 line comment")
	ast.FileExports(f)
	checkFieldComments(t, f, "T.F1", "/* F1 lead comment *///", "/* F1 */// line comment")
	checkFieldComments(t, f, "T.F2", "// F2 lead// comment", "// F2 line comment")
	if getField(f, "T.f3") != nil {
		t.Error("not expected to find T.f3")
	}
}

// TestIssue9979 verifies that empty statements are contained within their enclosing blocks.
func TestIssue9979(t *testing.T) {
	for _, src := range []string{
		"package p; func f() {;}",
		"package p; func f() {L:}",
		"package p; func f() {L:;}",
		"package p; func f() {L:\n}",
		"package p; func f() {L:\n;}",
		"package p; func f() { ; }",
		"package p; func f() { L: }",
		"package p; func f() { L: ; }",
		"package p; func f() { L: \n}",
		"package p; func f() { L: \n; }",
	} {
		fset := token.NewFileSet()
		f, err := ParseFile(fset, "", src, 0)
		if err != nil {
			t.Fatal(err)
		}

		var pos, end token.Pos
		ast.Inspect(f, func(x ast.Node) bool {
			switch s := x.(type) {
			case *ast.BlockStmt:
				pos, end = s.Pos()+1, s.End()-1 // exclude "{", "}"
			case *ast.LabeledStmt:
				pos, end = s.Pos()+2, s.End() // exclude "L:"
			case *ast.EmptyStmt:
				// check containment
				if s.Pos() < pos || s.End() > end {
					t.Errorf("%s: %T[%d, %d] not inside [%d, %d]", src, s, s.Pos(), s.End(), pos, end)
				}
				// check semicolon
				offs := fset.Position(s.Pos()).Offset
				if ch := src[offs]; ch != ';' != s.Implicit {
					want := "want ';'"
					if s.Implicit {
						want = "but ';' is implicit"
					}
					t.Errorf("%s: found %q at offset %d; %s", src, ch, offs, want)
				}
			}
			return true
		})
	}
}

// TestIncompleteSelection ensures that an incomplete selector
// expression is parsed as a (blank) *ast.SelectorExpr, not a
// *ast.BadExpr.
func TestIncompleteSelection(t *testing.T) {
	for _, src := range []string{
		"package p; var _ = fmt.",             // at EOF
		"package p; var _ = fmt.\ntype X int", // not at EOF
	} {
		fset := token.NewFileSet()
		f, err := ParseFile(fset, "", src, 0)
		if err == nil {
			t.Errorf("ParseFile(%s) succeeded unexpectedly", src)
			continue
		}

		const wantErr = "expected selector or type assertion"
		if !strings.Contains(err.Error(), wantErr) {
			t.Errorf("ParseFile returned wrong error %q, want %q", err, wantErr)
		}

		var sel *ast.SelectorExpr
		ast.Inspect(f, func(n ast.Node) bool {
			if n, ok := n.(*ast.SelectorExpr); ok {
				sel = n
			}
			return true
		})
		if sel == nil {
			t.Error("found no *ast.SelectorExpr")
			continue
		}
		const wantSel = "&{fmt _}"
		if fmt.Sprint(sel) != wantSel {
			t.Errorf("found selector %s, want %s", sel, wantSel)
			continue
		}
	}
}
