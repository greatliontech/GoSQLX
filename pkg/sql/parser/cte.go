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

// Package parser - cte.go
// Common Table Expression (CTE) parsing with recursive CTE support.

package parser

import (
	"fmt"

	goerrors "github.com/ajitpratap0/GoSQLX/pkg/errors"
	"github.com/ajitpratap0/GoSQLX/pkg/models"
	"github.com/ajitpratap0/GoSQLX/pkg/sql/ast"
	"github.com/ajitpratap0/GoSQLX/pkg/sql/keywords"
)

// WITH summary(region, total) AS (SELECT region, SUM(amount) FROM sales GROUP BY region) SELECT * FROM summary
func (p *Parser) parseWithStatement() (ast.Statement, error) {
	withPos := p.currentLocation()
	// Consume WITH
	p.advance()

	// Check for RECURSIVE keyword
	recursive := false
	if p.isType(models.TokenTypeRecursive) {
		recursive = true
		p.advance()
	}

	// Parse Common Table Expressions
	ctes := []*ast.CommonTableExpr{}

	for {
		cte, err := p.parseCommonTableExpr()
		if err != nil {
			return nil, goerrors.InvalidCTEError(
				fmt.Sprintf("error parsing CTE definition: %v", err),
				p.currentLocation(),
				"",
			)
		}
		ctes = append(ctes, cte)

		// Check for more CTEs (comma-separated)
		if p.isType(models.TokenTypeComma) {
			p.advance() // Consume comma
			continue
		}
		break
	}

	// Create WITH clause
	withClause := &ast.WithClause{
		Recursive: recursive,
		CTEs:      ctes,
		Pos:       withPos,
	}

	// Parse the main statement that follows the WITH clause
	mainStmt, err := p.parseMainStatementAfterWith()
	if err != nil {
		return nil, goerrors.InvalidCTEError(
			fmt.Sprintf("error parsing statement after WITH clause: %v", err),
			p.currentLocation(),
			"",
		)
	}

	// Attach WITH clause to the main statement
	switch stmt := mainStmt.(type) {
	case *ast.SelectStatement:
		stmt.With = withClause
		return stmt, nil
	case *ast.SetOperation:
		// A WITH clause prefixing a set-operation chain scopes its CTEs
		// over the ENTIRE chain (standard SQL: the WITH is part of the
		// query expression). N-arm chains are built left-associatively as
		// SetOperation{Left: SetOperation{...}, Right: SELECT}, so the
		// WITH belongs on the INNERMOST left SELECT — where the 2-arm form
		// already carries it. Walk the left spine to find it; checking only
		// the immediate Left silently dropped the clause on 3+-arm chains.
		cur := stmt
		for {
			if leftSelect, ok := cur.Left.(*ast.SelectStatement); ok {
				leftSelect.With = withClause
				break
			}
			nested, ok := cur.Left.(*ast.SetOperation)
			if !ok {
				// Unreachable today: parseSelectWithSetOperations builds
				// every operand directly from parseSelectStatement, so a
				// chain's left spine is SetOperations bottoming out at a
				// SELECT. Defensive — if a parenthesised query-expression
				// operand is ever added, dropping the WITH here would
				// re-introduce the silent-CTE-loss bug class.
				break
			}
			cur = nested
		}
		return stmt, nil
	case *ast.InsertStatement:
		stmt.With = withClause
		return stmt, nil
	case *ast.UpdateStatement:
		stmt.With = withClause
		return stmt, nil
	case *ast.DeleteStatement:
		stmt.With = withClause
		return stmt, nil
	default:
		return nil, goerrors.InvalidCTEError(
			fmt.Sprintf("WITH clause not supported with statement type: %T", stmt),
			p.currentLocation(),
			"",
		)
	}
}

// parseCommonTableExpr parses a single Common Table Expression.
// It handles CTE name, optional column list, AS keyword, and the CTE query in parentheses.
//
// Syntax: cte_name [(column_list)] AS (query)
func (p *Parser) parseCommonTableExpr() (*ast.CommonTableExpr, error) {
	// Check recursion depth to prevent stack overflow in recursive CTEs
	// This is critical since CTEs can call parseStatement which leads back to more CTEs
	p.depth++
	defer func() { p.depth-- }()

	if p.depth > MaxRecursionDepth {
		return nil, goerrors.InvalidCTEError(
			fmt.Sprintf("maximum recursion depth exceeded (%d) - CTE too deeply nested", MaxRecursionDepth),
			p.currentLocation(),
			"",
		)
	}

	// ClickHouse scalar CTE: WITH <expr> AS <name>, ...
	// Detected when the token after WITH is not an identifier, or is an
	// identifier not followed by AS/( (which would be a standard CTE).
	if p.dialect == string(keywords.DialectClickHouse) && !p.isIdentifier() {
		scalarExpr, err := p.parseExpression()
		if err != nil {
			return nil, err
		}
		if !p.isType(models.TokenTypeAs) {
			return nil, p.expectedError("AS after scalar CTE expression")
		}
		p.advance() // Consume AS
		if !p.isIdentifier() {
			return nil, p.expectedError("name after AS in scalar CTE")
		}
		scalarName := p.currentToken.Token.Value
		scalarPos := p.currentLocation()
		p.advance()
		return &ast.CommonTableExpr{
			Name:       scalarName,
			ScalarExpr: scalarExpr,
			Pos:        scalarPos,
		}, nil
	}

	// Parse CTE name (supports double-quoted identifiers)
	if !p.isIdentifier() {
		return nil, p.expectedError("CTE name")
	}
	cteNamePos := p.currentLocation()
	name := p.currentToken.Token.Value
	p.advance()

	// Parse optional column list
	var columns []string
	if p.isType(models.TokenTypeLParen) {
		p.advance() // Consume (

		for {
			if !p.isIdentifier() {
				return nil, p.expectedError("column name")
			}
			columns = append(columns, p.currentToken.Token.Value)
			p.advance()

			if p.isType(models.TokenTypeComma) {
				p.advance() // Consume comma
				continue
			}
			break
		}

		if !p.isType(models.TokenTypeRParen) {
			return nil, p.expectedError(")")
		}
		p.advance() // Consume )
	}

	// Parse AS keyword
	if !p.isType(models.TokenTypeAs) {
		return nil, p.expectedError("AS")
	}
	p.advance()

	// Parse optional MATERIALIZED / NOT MATERIALIZED
	// Syntax: AS [NOT] MATERIALIZED (query)
	var materialized *bool
	if p.isType(models.TokenTypeNot) {
		p.advance() // Consume NOT
		if !p.isType(models.TokenTypeMaterialized) {
			return nil, p.expectedError("MATERIALIZED after NOT")
		}
		p.advance() // Consume MATERIALIZED
		notMaterialized := false
		materialized = &notMaterialized
	} else if p.isType(models.TokenTypeMaterialized) {
		p.advance() // Consume MATERIALIZED
		isMaterialized := true
		materialized = &isMaterialized
	}

	// Parse the CTE query (must be in parentheses)
	if !p.isType(models.TokenTypeLParen) {
		return nil, p.expectedError("( before CTE query")
	}
	p.advance() // Consume (

	// Parse the inner statement - handle SELECT with set operations support
	var stmt ast.Statement
	var err error

	if p.isType(models.TokenTypeSelect) {
		// For SELECT statements, use parseSelectWithSetOperations to support UNION, EXCEPT, INTERSECT
		p.advance() // Consume SELECT
		stmt, err = p.parseSelectWithSetOperations()
	} else {
		// For other statement types (INSERT, UPDATE, DELETE, WITH), use parseStatement
		stmt, err = p.parseStatement()
	}

	if err != nil {
		return nil, goerrors.InvalidCTEError(
			fmt.Sprintf("error parsing CTE subquery: %v", err),
			p.currentLocation(),
			"",
		)
	}

	if !p.isType(models.TokenTypeRParen) {
		return nil, p.expectedError(") after CTE query")
	}
	p.advance() // Consume )

	return &ast.CommonTableExpr{
		Name:         name,
		Columns:      columns,
		Statement:    stmt,
		Materialized: materialized,
		Pos:          cteNamePos,
	}, nil
}

// parseMainStatementAfterWith parses the main statement after WITH clause.
// It supports SELECT, INSERT, UPDATE, and DELETE statements, routing them to the appropriate
// parsers while preserving set operation support for SELECT statements.
func (p *Parser) parseMainStatementAfterWith() (ast.Statement, error) {
	if p.isType(models.TokenTypeSelect) {
		p.advance() // Consume SELECT
		return p.parseSelectWithSetOperations()
	} else if p.isType(models.TokenTypeInsert) {
		p.advance() // Consume INSERT
		return p.parseInsertStatement()
	} else if p.isType(models.TokenTypeUpdate) {
		p.advance() // Consume UPDATE
		return p.parseUpdateStatement()
	} else if p.isType(models.TokenTypeDelete) {
		p.advance() // Consume DELETE
		return p.parseDeleteStatement()
	}
	return nil, p.expectedError("SELECT, INSERT, UPDATE, or DELETE after WITH")
}
