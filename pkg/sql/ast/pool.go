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

// This file implements comprehensive object pooling for all major AST node types
// using sync.Pool. The pooling system provides 60-80% memory reduction in production
// workloads and 95%+ pool hit rates with proper usage patterns.
//
// IMPORTANT: Always use defer when returning pooled objects to prevent leaks.
// See doc.go for complete pooling documentation and usage examples.
package ast

import (
	"sync"
	"sync/atomic"

	"github.com/ajitpratap0/GoSQLX/pkg/metrics"
)

// poolLeakCount counts expressions that exceeded PutExpression's iterative
// work-queue cap and were drained via the recursive fallback. Non-zero values
// mean the AST is pathologically large (>MaxWorkQueueSize nodes in a single
// cleanup) or the queue algorithm needs tuning. Exposed via PoolLeakCount().
var poolLeakCount uint64

// PoolLeakCount returns the number of times PutExpression's iterative cleanup
// exceeded MaxWorkQueueSize and fell back to recursive drain. A non-zero
// return does NOT indicate a leak — the recursive drain still releases every
// node — but it flags that the work-queue cap was hit. Used for diagnostics
// and by leak tests.
func PoolLeakCount() uint64 {
	return atomic.LoadUint64(&poolLeakCount)
}

// ResetPoolLeakCount zeroes the pool-leak counter. Test-only helper.
func ResetPoolLeakCount() {
	atomic.StoreUint64(&poolLeakCount, 0)
}

// poolGuard detects double-release of pooled AST nodes.
//
// The pooling system is built on a SINGLE-OWNER contract: every node has
// exactly one parent that releases it (directly via its typed Put
// function, or recursively via releaseStatement / PutExpression). The
// contract is violated when two wrappers share an aliased inner pointer —
// the second release would re-pool a node the pool may already have
// handed back to a different caller, silently corrupting both callers'
// ASTs with zero signal.
//
// Go's type system cannot express ownership transfer of an interface
// pointer, so the violation cannot be made unrepresentable; instead every
// pooled node embeds this guard and every release path checks it:
// a repeated release is REFUSED (the node is neither re-zeroed nor
// re-pooled, and its children are not re-walked) and counted in
// PoolDoubleReleaseCount. Get paths re-arm the guard.
//
// DETECTION LIMIT (ABA): the guard detects a repeated release only until
// the pool RE-ISSUES the node — Get re-arms the guard, after which a
// stale release by a previous owner is indistinguishable from the new
// owner's legitimate release. Full detection would require ownership
// tokens in the Put API (a breaking change); a reference count has the
// same hole (1→0, re-issue →1, stale put →0). In practice the filed
// hazard — two wrappers of one AST torn down together — has no
// interleaved Get and is caught deterministically; the re-issue race is
// pinned as a documented limit by
// TestDoubleReleaseAfterReissueIsUndetectable.
//
// Note the guard covers pooled NODE types. Internal scratch pools that
// hold non-node values (the *[]Expression slice pools) cannot carry a
// guard; they are package-internal and never aliased across wrappers.
type poolGuard struct {
	released atomic.Bool
}

// guardRelease reports whether the node may be released. The first call
// after construction or Get returns true and marks the node released; any
// repeated call is a double-release: it is counted and refused. The CAS
// makes detection sound across goroutines: when two owners race to
// release the same aliased node, exactly one wins and the other is
// counted — a plain bool would let both pass on an interleaved read.
func (g *poolGuard) guardRelease() bool {
	if !g.released.CompareAndSwap(false, true) {
		atomic.AddUint64(&poolDoubleReleaseCount, 1)
		return false
	}
	return true
}

// guardReset re-arms the guard when a node leaves the pool.
func (g *poolGuard) guardReset() {
	g.released.Store(false)
}

// releasable is implemented by every pooled node type via the embedded
// poolGuard; PutExpression's iterative dispatch uses it to guard nodes
// uniformly without per-case checks.
type releasable interface {
	guardRelease() bool
}

// poolDoubleReleaseCount counts refused releases of already-released
// nodes. Non-zero values mean a caller violated the single-owner release
// contract (most likely two wrappers aliasing one inner statement); the
// release was refused, so the pool is NOT corrupted, but the caller bug
// should be found and fixed. Exposed via PoolDoubleReleaseCount.
var poolDoubleReleaseCount uint64

// PoolDoubleReleaseCount returns the number of refused double-releases
// since process start (or the last ResetPoolDoubleReleaseCount). Each one
// is a caller bug: a pooled node was released twice, violating the
// single-owner contract documented on poolGuard.
//
// A refusal returns early, so any Put-side accounting (e.g. the named
// pool metrics in ReleaseAST) is intentionally skipped: Get/Put metric
// counts diverge by exactly this counter's value.
func PoolDoubleReleaseCount() uint64 {
	return atomic.LoadUint64(&poolDoubleReleaseCount)
}

// ResetPoolDoubleReleaseCount zeroes the double-release counter.
// Test-only helper.
func ResetPoolDoubleReleaseCount() {
	atomic.StoreUint64(&poolDoubleReleaseCount, 0)
}

// Pool configuration constants control cleanup behavior to prevent resource exhaustion.
const (
	// MaxCleanupDepth limits recursion depth to prevent stack overflow during cleanup.
	// Set to 100 based on typical SQL query complexity. Deeply nested expressions
	// use iterative cleanup instead of recursion.
	MaxCleanupDepth = 100

	// MaxWorkQueueSize limits the total number of nodes that the iterative
	// PutExpression cleanup loop will process before resizing protection kicks in.
	// Historically this was 1000 and cleanup silently stopped after that,
	// leaking every remaining node (hundreds per parse for large IN lists).
	//
	// The value is now 100_000, large enough to drain every realistic SQL AST
	// (even a 10k-element IN list or deeply nested CTE forest) in a single
	// pass. The work queue itself is bounded by the live AST size — nodes
	// are pointers already allocated — so this does not materially increase
	// peak memory vs. the AST that already exists.
	//
	// If the cap is ever hit, PutExpression falls back to a depth-limited
	// recursive drain (bounded by MaxCleanupDepth) for the remaining queue
	// so no pooled nodes are silently leaked. See PutExpression for details.
	MaxWorkQueueSize = 100_000
)

var (
	// DDL statement pools
	createTableStmtPool = sync.Pool{
		New: func() interface{} {
			return &CreateTableStatement{
				Columns:     make([]ColumnDef, 0, 4),
				Constraints: make([]TableConstraint, 0, 2),
				Inherits:    make([]string, 0),
				Options:     make([]TableOption, 0),
			}
		},
	}

	alterTableStmtPool = sync.Pool{
		New: func() interface{} {
			return &AlterTableStatement{
				Actions: make([]AlterTableAction, 0, 2),
			}
		},
	}

	createIndexStmtPool = sync.Pool{
		New: func() interface{} {
			return &CreateIndexStatement{
				Columns: make([]IndexColumn, 0, 4),
			}
		},
	}

	mergeStmtPool = sync.Pool{
		New: func() interface{} {
			return &MergeStatement{
				WhenClauses: make([]*MergeWhenClause, 0, 2),
				Output:      make([]Expression, 0, 2),
			}
		},
	}

	createViewStmtPool = sync.Pool{
		New: func() interface{} {
			return &CreateViewStatement{
				Columns: make([]string, 0),
			}
		},
	}

	createMaterializedViewStmtPool = sync.Pool{
		New: func() interface{} {
			return &CreateMaterializedViewStatement{
				Columns: make([]string, 0),
			}
		},
	}

	refreshMaterializedViewStmtPool = sync.Pool{
		New: func() interface{} {
			return &RefreshMaterializedViewStatement{}
		},
	}

	dropStmtPool = sync.Pool{
		New: func() interface{} {
			return &DropStatement{
				Names: make([]string, 0, 2),
			}
		},
	}

	truncateStmtPool = sync.Pool{
		New: func() interface{} {
			return &TruncateStatement{
				Tables: make([]string, 0, 2),
			}
		},
	}

	showStmtPool = sync.Pool{
		New: func() interface{} {
			return &ShowStatement{}
		},
	}

	describeStmtPool = sync.Pool{
		New: func() interface{} {
			return &DescribeStatement{}
		},
	}

	explainStmtPool = sync.Pool{
		New: func() interface{} {
			return &ExplainStatement{}
		},
	}

	unsupportedStmtPool = sync.Pool{
		New: func() interface{} {
			return &UnsupportedStatement{}
		},
	}

	replaceStmtPool = sync.Pool{
		New: func() interface{} {
			return &ReplaceStatement{
				Columns: make([]Expression, 0, 4),
				Values:  make([][]Expression, 0, 4),
			}
		},
	}

	alterStmtPool = sync.Pool{
		New: func() interface{} {
			return &AlterStatement{}
		},
	}

	// AST node pools
	astPool = sync.Pool{
		New: func() interface{} {
			return &AST{
				Statements: make([]Statement, 0, 8), // Increased initial capacity
			}
		},
	}

	// Statement pools
	selectStmtPool = sync.Pool{
		New: func() interface{} {
			return &SelectStatement{
				Columns: make([]Expression, 0, 4),
				OrderBy: make([]OrderByExpression, 0, 1),
			}
		},
	}

	insertStmtPool = sync.Pool{
		New: func() interface{} {
			return &InsertStatement{
				Columns: make([]Expression, 0, 4),
				Values:  make([][]Expression, 0, 4),
			}
		},
	}

	updateStmtPool = sync.Pool{
		New: func() interface{} {
			return &UpdateStatement{
				Assignments: make([]UpdateExpression, 0, 4),
			}
		},
	}

	deleteStmtPool = sync.Pool{
		New: func() interface{} {
			return &DeleteStatement{}
		},
	}

	// Expression pools
	identifierPool = sync.Pool{
		New: func() interface{} {
			return &Identifier{}
		},
	}

	binaryExprPool = sync.Pool{
		New: func() interface{} {
			return &BinaryExpression{}
		},
	}

	// Add a pool for LiteralValue to reduce allocations
	literalValuePool = sync.Pool{
		New: func() interface{} {
			return &LiteralValue{}
		},
	}

	updateExprPool = sync.Pool{
		New: func() interface{} {
			return &UpdateExpression{}
		},
	}

	// Additional expression pools for common expression types
	functionCallPool = sync.Pool{
		New: func() interface{} {
			return &FunctionCall{
				Arguments: make([]Expression, 0, 4),
			}
		},
	}

	caseExprPool = sync.Pool{
		New: func() interface{} {
			return &CaseExpression{
				WhenClauses: make([]WhenClause, 0, 2),
			}
		},
	}

	betweenExprPool = sync.Pool{
		New: func() interface{} {
			return &BetweenExpression{}
		},
	}

	inExprPool = sync.Pool{
		New: func() interface{} {
			return &InExpression{
				List: make([]Expression, 0, 4),
			}
		},
	}

	tupleExprPool = sync.Pool{
		New: func() interface{} {
			return &TupleExpression{
				Expressions: make([]Expression, 0, 4),
			}
		},
	}

	arrayConstructorPool = sync.Pool{
		New: func() interface{} {
			return &ArrayConstructorExpression{
				Elements: make([]Expression, 0, 4),
			}
		},
	}

	subqueryExprPool = sync.Pool{
		New: func() interface{} {
			return &SubqueryExpression{}
		},
	}

	castExprPool = sync.Pool{
		New: func() interface{} {
			return &CastExpression{}
		},
	}

	intervalExprPool = sync.Pool{
		New: func() interface{} {
			return &IntervalExpression{}
		},
	}

	arraySubscriptExprPool = sync.Pool{
		New: func() interface{} {
			return &ArraySubscriptExpression{
				Indices: make([]Expression, 0, 2), // Most common: 1-2 dimensions
			}
		},
	}

	arraySliceExprPool = sync.Pool{
		New: func() interface{} {
			return &ArraySliceExpression{}
		},
	}

	// Additional expression pools for complete coverage
	existsExprPool = sync.Pool{
		New: func() interface{} {
			return &ExistsExpression{}
		},
	}

	anyExprPool = sync.Pool{
		New: func() interface{} {
			return &AnyExpression{}
		},
	}

	allExprPool = sync.Pool{
		New: func() interface{} {
			return &AllExpression{}
		},
	}

	listExprPool = sync.Pool{
		New: func() interface{} {
			return &ListExpression{
				Values: make([]Expression, 0, 4),
			}
		},
	}

	unaryExprPool = sync.Pool{
		New: func() interface{} {
			return &UnaryExpression{}
		},
	}

	extractExprPool = sync.Pool{
		New: func() interface{} {
			return &ExtractExpression{}
		},
	}

	positionExprPool = sync.Pool{
		New: func() interface{} {
			return &PositionExpression{}
		},
	}

	substringExprPool = sync.Pool{
		New: func() interface{} {
			return &SubstringExpression{}
		},
	}

	aliasedExprPool = sync.Pool{
		New: func() interface{} {
			return &AliasedExpression{}
		},
	}

	// Slice pools
	exprSlicePool = sync.Pool{
		New: func() interface{} {
			s := make([]Expression, 0, 16) // Double capacity for better performance
			return &s
		},
	}

	createSequencePool = sync.Pool{
		New: func() interface{} { return &CreateSequenceStatement{} },
	}

	dropSequencePool = sync.Pool{
		New: func() interface{} { return &DropSequenceStatement{} },
	}

	alterSequencePool = sync.Pool{
		New: func() interface{} { return &AlterSequenceStatement{} },
	}

	// putExpressionWorkQueuePool recycles the iterative work-queue slice used
	// by putExpressionImpl. Pre-fix, putExpressionImpl allocated a fresh
	// []Expression with cap 32 on every call — that fires 10-100× per parse
	// in hot paths (complex SELECTs, deep expression trees), contributing
	// measurable alloc-rate and GC pressure to an otherwise zero-copy hot
	// path. Pooling the queue reclaims the allocation.
	//
	// Storing a *[]Expression (not []Expression) avoids the slice-header
	// boxing allocation that happens when you store a slice in an interface.
	// Callers must write the mutated slice header back to the pointer
	// before Put so subsequent Get sees the grown capacity.
	putExpressionWorkQueuePool = sync.Pool{
		New: func() interface{} {
			s := make([]Expression, 0, 32)
			return &s
		},
	}
)

// NewAST retrieves a new AST container from the pool.
//
// NewAST returns a pooled AST container with pre-allocated statement capacity.
// This is the primary entry point for creating AST objects with memory pooling.
//
// Usage Pattern (MANDATORY):
//
//	astObj := ast.NewAST()
//	defer ast.ReleaseAST(astObj)  // ALWAYS use defer to prevent leaks
//
//	// Use astObj...
//
// The returned AST has:
//   - Empty Statements slice with capacity for 8 statements
//   - Clean state ready for population
//
// Performance:
//   - 95%+ pool hit rate in production workloads
//   - Eliminates allocation overhead for AST containers
//   - Reduces GC pressure by reusing objects
//
// CRITICAL: Always call ReleaseAST() when done, preferably via defer.
// Failure to return objects to the pool causes memory leaks and degrades
// performance by forcing new allocations.
//
// Example:
//
//	func parseQuery(sql string) (*ast.AST, error) {
//	    astObj := ast.NewAST()
//	    defer ast.ReleaseAST(astObj)
//
//	    // Parse and populate AST
//	    stmt := ast.GetSelectStatement()
//	    defer ast.PutSelectStatement(stmt)
//	    // ... build statement ...
//	    astObj.Statements = append(astObj.Statements, stmt)
//
//	    return astObj, nil
//	}
//
// See also: ReleaseAST(), GetSelectStatement(), GetInsertStatement()
func NewAST() *AST {
	metrics.RecordNamedPoolGet("ast")
	x := astPool.Get().(*AST)
	x.guardReset()
	return x
}

// ReleaseAST returns an AST container to the pool for reuse.
//
// ReleaseAST cleans up and returns the AST to the pool, allowing it to be
// reused in future NewAST() calls. This is critical for memory efficiency
// and performance.
//
// Cleanup Process:
//  1. Returns all statement objects to their respective pools
//  2. Clears all statement references
//  3. Resets the Statements slice (preserves capacity)
//  4. Returns the AST container to astPool
//
// Usage Pattern (MANDATORY):
//
//	astObj := ast.NewAST()
//	defer ast.ReleaseAST(astObj)  // ALWAYS use defer
//
// Parameters:
//   - ast: AST container to return (nil-safe, ignores nil)
//
// The function is nil-safe and will return immediately if passed a nil AST.
//
// CRITICAL: This function must be called for every AST obtained from NewAST().
// Use defer immediately after NewAST() to ensure cleanup even on error paths.
//
// Performance Impact:
//   - Prevents memory leaks by returning objects to pools
//   - Maintains 95%+ pool hit rates
//   - Reduces GC overhead by reusing allocations
//   - Essential for sustained high throughput (1.38M+ ops/sec)
//
// Example - Correct usage:
//
//	func processSQL(sql string) error {
//	    astObj := ast.NewAST()
//	    defer ast.ReleaseAST(astObj)  // Cleanup guaranteed
//
//	    // ... process astObj ...
//	    return nil
//	}
//
// See also: NewAST(), PutSelectStatement(), PutInsertStatement()
func ReleaseAST(ast *AST) {
	if ast == nil {
		return
	}
	if !ast.guardRelease() {
		return
	}

	// Clean up all statements
	for i := range ast.Statements {
		releaseStatement(ast.Statements[i])
		ast.Statements[i] = nil
	}

	// Reset slice but keep capacity
	ast.Statements = ast.Statements[:0]

	// Reset comments but keep capacity
	if cap(ast.Comments) > 0 {
		ast.Comments = ast.Comments[:0]
	}

	// Return to pool
	metrics.RecordNamedPoolPut("ast")
	astPool.Put(ast)
}

// ReleaseStatements returns a slice of statements back to their respective pools.
// Use this to clean up statements returned by ParseWithRecovery, which returns
// []Statement rather than an *AST.
//
// Example:
//
//	stmts, errs := parser.ParseWithRecovery(tokens)
//	defer ast.ReleaseStatements(stmts)
//	// ... process stmts and errs ...
func ReleaseStatements(stmts []Statement) {
	for i := range stmts {
		if stmts[i] == nil {
			continue
		}
		releaseStatement(stmts[i])
		stmts[i] = nil
	}
}

// releaseStatement returns a single Statement to its pool.
// This is the central dispatch used by both ReleaseAST and ReleaseStatements.
func releaseStatement(stmt Statement) {
	if stmt == nil {
		return
	}
	switch s := stmt.(type) {
	case *SelectStatement:
		PutSelectStatement(s)
	case *InsertStatement:
		PutInsertStatement(s)
	case *UpdateStatement:
		PutUpdateStatement(s)
	case *DeleteStatement:
		PutDeleteStatement(s)
	case *SetOperation:
		// A set operation owns its two operands; release both so their
		// pooled sub-nodes (expressions, CTE bodies, subqueries) return to
		// their pools. N-arm chains nest on the left spine, so this
		// recurses the whole chain. SetOperation is heap-allocated (not
		// pooled) — release the children and drop the links. Without this
		// case every set-op query silently leaked its operands' pooled
		// nodes (the same un-dispatched-pooled-statement class fixed above
		// for the sequence statements).
		releaseStatement(s.Left)
		releaseStatement(s.Right)
		s.Left = nil
		s.Right = nil
	case *CreateTableStatement:
		PutCreateTableStatement(s)
	case *AlterTableStatement:
		PutAlterTableStatement(s)
	case *CreateIndexStatement:
		PutCreateIndexStatement(s)
	case *MergeStatement:
		PutMergeStatement(s)
	case *CreateViewStatement:
		PutCreateViewStatement(s)
	case *CreateMaterializedViewStatement:
		PutCreateMaterializedViewStatement(s)
	case *RefreshMaterializedViewStatement:
		PutRefreshMaterializedViewStatement(s)
	case *DropStatement:
		PutDropStatement(s)
	case *TruncateStatement:
		PutTruncateStatement(s)
	case *ShowStatement:
		PutShowStatement(s)
	case *DescribeStatement:
		PutDescribeStatement(s)
	case *ExplainStatement:
		PutExplainStatement(s)
	case *UnsupportedStatement:
		PutUnsupportedStatement(s)
	case *ReplaceStatement:
		PutReplaceStatement(s)
	case *AlterStatement:
		PutAlterStatement(s)
	// Sequence statements are pooled via NewXxx/ReleaseXxx helpers.
	// Without a dispatch here, a CTE or subquery that contained one
	// would silently leak it (the stmt type audit found these three
	// pooled but un-dispatched; see architect review sprint 2).
	case *CreateSequenceStatement:
		ReleaseCreateSequenceStatement(s)
	case *DropSequenceStatement:
		ReleaseDropSequenceStatement(s)
	case *AlterSequenceStatement:
		ReleaseAlterSequenceStatement(s)
		// NOTE: *PragmaStatement is NOT pooled (no sync.Pool declared);
		// intentionally no-op. Same for dml.go's *Select/*Insert/*Update/
		// *Delete (legacy unpooled duplicates) — they'd be GC'd naturally.
		// If those types ever gain pools, add cases here.
	}
}

// GetCreateIndexStatement gets a CreateIndexStatement from the pool.
func GetCreateIndexStatement() *CreateIndexStatement {
	stmt := createIndexStmtPool.Get().(*CreateIndexStatement)
	stmt.guardReset()
	stmt.Columns = stmt.Columns[:0]
	return stmt
}

// PutCreateIndexStatement returns a CreateIndexStatement to the pool.
// It releases the optional WHERE expression.
func PutCreateIndexStatement(stmt *CreateIndexStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	PutExpression(stmt.Where)

	for i := range stmt.Columns {
		stmt.Columns[i].Column = ""
		stmt.Columns[i].Collate = ""
		stmt.Columns[i].Direction = ""
		stmt.Columns[i].NullsLast = false
	}
	stmt.Columns = stmt.Columns[:0]

	stmt.Where = nil
	stmt.Unique = false
	stmt.IfNotExists = false
	stmt.Name = ""
	stmt.Table = ""
	stmt.Using = ""

	createIndexStmtPool.Put(stmt)
}

// GetCreateViewStatement gets a CreateViewStatement from the pool.
func GetCreateViewStatement() *CreateViewStatement {
	stmt := createViewStmtPool.Get().(*CreateViewStatement)
	stmt.guardReset()
	stmt.Columns = stmt.Columns[:0]
	return stmt
}

// PutCreateViewStatement returns a CreateViewStatement to the pool.
// It recursively releases the nested query statement.
func PutCreateViewStatement(stmt *CreateViewStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	// Recursively release the nested SELECT query
	releaseStatement(stmt.Query)

	stmt.OrReplace = false
	stmt.Temporary = false
	stmt.IfNotExists = false
	stmt.Name = ""
	stmt.Columns = stmt.Columns[:0]
	stmt.Query = nil
	stmt.WithOption = ""

	createViewStmtPool.Put(stmt)
}

// GetCreateMaterializedViewStatement gets a CreateMaterializedViewStatement from the pool.
func GetCreateMaterializedViewStatement() *CreateMaterializedViewStatement {
	stmt := createMaterializedViewStmtPool.Get().(*CreateMaterializedViewStatement)
	stmt.guardReset()
	stmt.Columns = stmt.Columns[:0]
	return stmt
}

// PutCreateMaterializedViewStatement returns a CreateMaterializedViewStatement to the pool.
// It recursively releases the nested query statement.
func PutCreateMaterializedViewStatement(stmt *CreateMaterializedViewStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	// Recursively release the nested SELECT query
	releaseStatement(stmt.Query)

	stmt.IfNotExists = false
	stmt.Name = ""
	stmt.Columns = stmt.Columns[:0]
	stmt.Query = nil
	stmt.WithData = nil
	stmt.Tablespace = ""

	createMaterializedViewStmtPool.Put(stmt)
}

// GetRefreshMaterializedViewStatement gets a RefreshMaterializedViewStatement from the pool.
func GetRefreshMaterializedViewStatement() *RefreshMaterializedViewStatement {
	x := refreshMaterializedViewStmtPool.Get().(*RefreshMaterializedViewStatement)
	x.guardReset()
	return x
}

// PutRefreshMaterializedViewStatement returns a RefreshMaterializedViewStatement to the pool.
func PutRefreshMaterializedViewStatement(stmt *RefreshMaterializedViewStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.Concurrently = false
	stmt.Name = ""
	stmt.WithData = nil

	refreshMaterializedViewStmtPool.Put(stmt)
}

// GetDropStatement gets a DropStatement from the pool.
func GetDropStatement() *DropStatement {
	stmt := dropStmtPool.Get().(*DropStatement)
	stmt.guardReset()
	stmt.Names = stmt.Names[:0]
	return stmt
}

// PutDropStatement returns a DropStatement to the pool.
func PutDropStatement(stmt *DropStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.ObjectType = ""
	stmt.IfExists = false
	stmt.Names = stmt.Names[:0]
	stmt.CascadeType = ""

	dropStmtPool.Put(stmt)
}

// GetTruncateStatement gets a TruncateStatement from the pool.
func GetTruncateStatement() *TruncateStatement {
	stmt := truncateStmtPool.Get().(*TruncateStatement)
	stmt.guardReset()
	stmt.Tables = stmt.Tables[:0]
	return stmt
}

// PutTruncateStatement returns a TruncateStatement to the pool.
func PutTruncateStatement(stmt *TruncateStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.Tables = stmt.Tables[:0]
	stmt.RestartIdentity = false
	stmt.ContinueIdentity = false
	stmt.CascadeType = ""

	truncateStmtPool.Put(stmt)
}

// GetShowStatement gets a ShowStatement from the pool.
func GetShowStatement() *ShowStatement {
	x := showStmtPool.Get().(*ShowStatement)
	x.guardReset()
	return x
}

// PutShowStatement returns a ShowStatement to the pool.
func PutShowStatement(stmt *ShowStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.ShowType = ""
	stmt.ObjectName = ""
	stmt.From = ""

	showStmtPool.Put(stmt)
}

// GetDescribeStatement gets a DescribeStatement from the pool.
func GetDescribeStatement() *DescribeStatement {
	x := describeStmtPool.Get().(*DescribeStatement)
	x.guardReset()
	return x
}

// PutDescribeStatement returns a DescribeStatement to the pool.
func PutDescribeStatement(stmt *DescribeStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.TableName = ""

	describeStmtPool.Put(stmt)
}

// GetExplainStatement gets an ExplainStatement from the pool.
func GetExplainStatement() *ExplainStatement {
	x := explainStmtPool.Get().(*ExplainStatement)
	x.guardReset()
	return x
}

// PutExplainStatement returns an ExplainStatement to the pool.
// Recursively releases the nested inner Statement — matches the
// wrapper-owns-inner convention shared with PutCreateViewStatement,
// PutCreateMaterializedViewStatement, and PutInsertStatement. Callers
// must not retain aliases to the inner after calling this.
func PutExplainStatement(stmt *ExplainStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	releaseStatement(stmt.Statement)
	stmt.Statement = nil
	stmt.Analyze = false
	stmt.Format = ""
	stmt.Mode = ""

	explainStmtPool.Put(stmt)
}

// GetUnsupportedStatement gets an UnsupportedStatement from the pool.
func GetUnsupportedStatement() *UnsupportedStatement {
	x := unsupportedStmtPool.Get().(*UnsupportedStatement)
	x.guardReset()
	return x
}

// PutUnsupportedStatement returns an UnsupportedStatement to the pool.
func PutUnsupportedStatement(stmt *UnsupportedStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.Kind = ""
	stmt.RawSQL = ""

	unsupportedStmtPool.Put(stmt)
}

// GetAlterStatement gets an AlterStatement from the pool.
func GetAlterStatement() *AlterStatement {
	x := alterStmtPool.Get().(*AlterStatement)
	x.guardReset()
	return x
}

// PutAlterStatement returns an AlterStatement to the pool.
// It zeroes all fields; the Operation interface value is cleared but
// its internal allocations are not recursively pooled (they use custom types).
func PutAlterStatement(stmt *AlterStatement) {
	if stmt == nil {
		return
	}
	if !stmt.guardRelease() {
		return
	}

	stmt.Type = 0
	stmt.Name = ""
	stmt.Operation = nil

	alterStmtPool.Put(stmt)
}

// NewCreateSequenceStatement retrieves a CreateSequenceStatement from the pool.
func NewCreateSequenceStatement() *CreateSequenceStatement {
	x := createSequencePool.Get().(*CreateSequenceStatement)
	x.guardReset()
	return x
}

// ReleaseCreateSequenceStatement returns a CreateSequenceStatement to the pool.
func ReleaseCreateSequenceStatement(s *CreateSequenceStatement) {
	if s == nil || !s.guardRelease() {
		return
	}
	// Release the pooled name identifier before the wipe drops it.
	// (Concrete-type nil check: a typed nil inside the Expression
	// interface would not compare equal to nil in PutExpression.)
	if s.Name != nil {
		PutExpression(s.Name)
	}
	// Zero all fields, then re-mark released: the whole-struct wipe
	// resets the embedded guard, and a pooled node must read as
	// released until Get re-arms it.
	*s = CreateSequenceStatement{}
	s.released.Store(true)
	createSequencePool.Put(s)
}

// NewDropSequenceStatement retrieves a DropSequenceStatement from the pool.
func NewDropSequenceStatement() *DropSequenceStatement {
	x := dropSequencePool.Get().(*DropSequenceStatement)
	x.guardReset()
	return x
}

// ReleaseDropSequenceStatement returns a DropSequenceStatement to the pool.
// Always call this with defer after parsing is complete.
func ReleaseDropSequenceStatement(s *DropSequenceStatement) {
	if s == nil || !s.guardRelease() {
		return
	}
	// Release the pooled name identifier before the wipe drops it.
	// (Concrete-type nil check: a typed nil inside the Expression
	// interface would not compare equal to nil in PutExpression.)
	if s.Name != nil {
		PutExpression(s.Name)
	}
	// Zero all fields, then re-mark released: the whole-struct wipe
	// resets the embedded guard, and a pooled node must read as
	// released until Get re-arms it.
	*s = DropSequenceStatement{}
	s.released.Store(true)
	dropSequencePool.Put(s)
}

// NewAlterSequenceStatement retrieves an AlterSequenceStatement from the pool.
func NewAlterSequenceStatement() *AlterSequenceStatement {
	x := alterSequencePool.Get().(*AlterSequenceStatement)
	x.guardReset()
	return x
}

// ReleaseAlterSequenceStatement returns an AlterSequenceStatement to the pool.
func ReleaseAlterSequenceStatement(s *AlterSequenceStatement) {
	if s == nil || !s.guardRelease() {
		return
	}
	// Release the pooled name identifier before the wipe drops it.
	// (Concrete-type nil check: a typed nil inside the Expression
	// interface would not compare equal to nil in PutExpression.)
	if s.Name != nil {
		PutExpression(s.Name)
	}
	// Zero all fields, then re-mark released: the whole-struct wipe
	// resets the embedded guard, and a pooled node must read as
	// released until Get re-arms it.
	*s = AlterSequenceStatement{}
	s.released.Store(true)
	alterSequencePool.Put(s)
}
