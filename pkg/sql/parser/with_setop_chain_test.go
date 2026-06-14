// Copyright 2026 GoSQLX Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package parser

import (
	"testing"

	"github.com/ajitpratap0/GoSQLX/pkg/sql/ast"
	"github.com/ajitpratap0/GoSQLX/pkg/sql/tokenizer"
)

// A WITH clause prefixing a set-operation chain scopes its CTEs over the
// ENTIRE chain (standard SQL: the CTE is part of the query expression).
// The parser builds N-arm chains left-associatively as
// SetOperation{Left: SetOperation{...}, Right: SELECT}, so the WITH must
// be attached to the INNERMOST left SELECT — where the 2-arm form already
// puts it. Before the fix, only the 2-arm form retained the WITH; 3+-arm
// chains dropped it (silent CTE loss).
func TestParser_WithRetainedOnMultiArmSetOpChain(t *testing.T) {
	cases := []struct {
		name string
		sql  string
	}{
		{"two_arm", `WITH x AS (SELECT 1 AS n) SELECT * FROM x UNION ALL SELECT 2`},
		{"three_arm", `WITH x AS (SELECT 1 AS n) SELECT * FROM x UNION ALL SELECT 2 UNION ALL SELECT 3`},
		{"four_arm_mixed", `WITH x AS (SELECT 1 AS n) SELECT * FROM x UNION SELECT 2 INTERSECT SELECT 3 EXCEPT SELECT 4`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tkz := tokenizer.GetTokenizer()
			defer tokenizer.PutTokenizer(tkz)
			tokens, err := tkz.Tokenize([]byte(tc.sql))
			if err != nil {
				t.Fatalf("tokenize: %v", err)
			}
			parser := &Parser{}
			astObj, err := parser.ParseFromModelTokens(tokens)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			defer ast.ReleaseAST(astObj)

			setOp, ok := astObj.Statements[0].(*ast.SetOperation)
			if !ok {
				t.Fatalf("expected SetOperation, got %T", astObj.Statements[0])
			}

			// Walk the left spine to the innermost left SELECT.
			var sel *ast.SelectStatement
			cur := setOp
			for {
				if s, ok := cur.Left.(*ast.SelectStatement); ok {
					sel = s
					break
				}
				nested, ok := cur.Left.(*ast.SetOperation)
				if !ok {
					t.Fatalf("left spine bottomed out at %T, not a SELECT", cur.Left)
				}
				cur = nested
			}

			if sel.With == nil {
				t.Fatal("WITH clause dropped from the set-op chain (CTE silently lost)")
			}
			if len(sel.With.CTEs) != 1 || sel.With.CTEs[0].Name != "x" {
				t.Fatalf("unexpected WITH contents: %+v", sel.With)
			}
		})
	}
}

// Releasing a set-operation chain must return its operand SELECTs — and
// their nested pooled nodes — to the pools. releaseStatement now
// dispatches *SetOperation; without it every set-op query silently leaked
// its operands (releaseStatement fell through to a no-op).
func TestReleaseAST_SetOpChainReleasesOperands(t *testing.T) {
	cases := []string{
		`SELECT 1 UNION ALL SELECT 2`,
		`SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3`,
		`WITH x AS (SELECT 1 AS n) SELECT * FROM x UNION ALL SELECT 2 UNION ALL SELECT 3`,
		`WITH y AS (SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) SELECT * FROM y`, // set-op as CTE body
		// The same dispatcher fixes set-ops embedded elsewhere — every
		// holder of a possibly-set-op statement routes through
		// releaseStatement: IN/EXISTS subqueries and INSERT ... SELECT.
		// These were leaking pre-fix too (top statement isn't a chain, so
		// the guard-probe is skipped; the no-double-release check covers
		// them).
		`SELECT * FROM t WHERE x IN (SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3)`,
		`SELECT * FROM t WHERE EXISTS (SELECT 1 UNION ALL SELECT 2)`,
		`INSERT INTO t SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3`,
	}
	for _, sql := range cases {
		t.Run(sql, func(t *testing.T) {
			ast.ResetPoolDoubleReleaseCount()

			tkz := tokenizer.GetTokenizer()
			tokens, err := tkz.Tokenize([]byte(sql))
			tokenizer.PutTokenizer(tkz)
			if err != nil {
				t.Fatalf("tokenize: %v", err)
			}
			parser := &Parser{}
			astObj, err := parser.ParseFromModelTokens(tokens)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}

			// Capture the innermost left operand SELECT when the top
			// statement is a chain, so we can prove it was released.
			var captured *ast.SelectStatement
			if setOp, ok := astObj.Statements[0].(*ast.SetOperation); ok {
				cur := setOp
				for {
					if s, ok := cur.Left.(*ast.SelectStatement); ok {
						captured = s
						break
					}
					n, ok := cur.Left.(*ast.SetOperation)
					if !ok {
						break
					}
					cur = n
				}
			}

			ast.ReleaseAST(astObj)
			if c := ast.PoolDoubleReleaseCount(); c != 0 {
				t.Fatalf("ReleaseAST double-released %d node(s)", c)
			}

			// Prove the operand was actually released: a second release is
			// REFUSED (the guard is armed). On the pre-fix code the operand
			// was never released, so this would SUCCEED and the count stay
			// 0 — exactly the leak the dispatch fixes.
			if captured != nil {
				ast.ResetPoolDoubleReleaseCount()
				ast.PutSelectStatement(captured)
				if ast.PoolDoubleReleaseCount() == 0 {
					t.Fatal("operand SELECT was NOT released by ReleaseAST (leak)")
				}
			}
		})
	}
}
