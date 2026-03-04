# Feature: Phase 0 — Build Coordinator

## Summary
A single orchestration document that, when kicked off, autonomously builds the entire ferrolearn library across four phases by spawning, monitoring, and merging the work of ~33 subagents. The coordinator is the only thing the human launches. It reads the phase design docs, spawns agents in dependency order, gates phase transitions on acceptance criteria, recovers stuck agents, and merges worktrees into a coherent codebase. The human's role reduces to approving phase gates and resolving any issues the coordinator escalates.

## Requirements

### Orchestration Core

- REQ-1: The coordinator reads `.design/phase-{1,2,3,4}-*.md` as its authoritative task definitions — it does not invent work beyond what the design docs specify
- REQ-2: Agents are spawned using the `Agent` tool with `isolation: "worktree"` so each works on an isolated copy of the repo
- REQ-3: The coordinator maintains a dependency DAG and never spawns an agent before its dependencies have been merged to the integration branch
- REQ-4: Phase transitions (1→2, 2→3, 3→4) are gated on all acceptance criteria for the completing phase passing — verified by running `cargo test --workspace` and `cargo clippy --workspace` on the integration branch
- REQ-5: The coordinator uses crosslink issues to track every agent's assignment, status, and outcome
- REQ-6: When an agent completes, the coordinator merges its worktree branch into the `dev` integration branch, resolving conflicts if necessary
- REQ-7: When an agent is stuck (no progress for 3+ turns, or reports a blocker), the coordinator reads the agent's output, diagnoses the issue, and either provides guidance by resuming the agent or spawns a fixup agent
- REQ-8: After each phase completes, the coordinator runs the full test suite and records the result as a crosslink comment before proceeding
- REQ-9: The coordinator creates a `CLAUDE.md` project file at the start with conventions all subagents must follow (import paths, error handling patterns, formatting, test patterns)
- REQ-10: On context compression, the coordinator recovers by re-reading crosslink issue state and the current phase design doc to reconstruct its position

### Agent Management

- REQ-11: Each spawned agent receives a prompt containing: (a) the specific requirements it owns, (b) the file paths it should create/modify, (c) the acceptance criteria it must satisfy, (d) the dependency versions and import conventions from `CLAUDE.md`, and (e) explicit instructions to commit its work and run `cargo test` before finishing
- REQ-12: Agents are spawned with `model: "sonnet"` for straightforward implementation tasks and `model: "opus"` for architecturally complex tasks (custom L-BFGS, Backend trait, compile-time pipeline, GPU backends)
- REQ-13: The coordinator tracks agent IDs and uses `resume` to continue conversations with agents that need guidance
- REQ-14: Maximum 8 concurrent background agents to avoid resource exhaustion
- REQ-15: Each agent's crosslink issue includes a `--kind result` comment documenting what was delivered, what tests pass, and any known issues

### Merge Strategy

- REQ-16: An integration branch `dev` is created from `main` at the start — all agent work merges here
- REQ-17: Merges happen sequentially (not concurrent) to avoid compound conflicts
- REQ-18: After each merge, the coordinator runs `cargo build --workspace` to verify compilation — if it fails, the coordinator spawns a fixup agent targeting the conflict
- REQ-19: After all agents in a phase are merged, the coordinator runs the full acceptance criteria check before starting the next phase
- REQ-20: At the end of Phase 4, the coordinator creates a PR from `dev` to `main` with a summary of all work

## Acceptance Criteria

- [ ] AC-1: Running `crosslink kickoff run "Build ferrolearn" --doc .design/phase-0-coordinator.md` produces a working library with no further human intervention (beyond phase gate approvals)
- [ ] AC-2: All 55 acceptance criteria across phases 1-4 pass on the final `dev` branch
- [ ] AC-3: Every agent's work is tracked via a crosslink issue with typed comments
- [ ] AC-4: No agent is left in a stuck state for more than 10 minutes without coordinator intervention
- [ ] AC-5: The `dev` branch has a clean, linear-ish commit history with meaningful commit messages
- [ ] AC-6: `cargo test --workspace` passes after each phase gate
- [ ] AC-7: Context compression does not cause the coordinator to lose track of progress or duplicate work

## Architecture

### Coordinator State Machine

```
START
  │
  ├─► Create `dev` branch from `main`
  ├─► Write `CLAUDE.md` with project conventions
  ├─► Create crosslink epic "Build ferrolearn" with sub-issues per agent
  │
  ▼
PHASE_1_CORE
  │
  ├─► Spawn Agent 1: ferrolearn-core (BLOCKING — everything depends on this)
  ├─► Wait for Agent 1 completion
  ├─► Merge Agent 1 worktree → dev
  ├─► Verify: cargo build --workspace
  │
  ▼
PHASE_1_PARALLEL
  │
  ├─► Spawn Agents 2-7 in parallel (all depend only on core)
  │     Agent 2: ferrolearn-sparse
  │     Agent 3: ferrolearn-metrics
  │     Agent 4: ferrolearn-preprocess
  │     Agent 5: ferrolearn-linear (L-BFGS — opus model)
  │     Agent 6: ferrolearn-model-sel
  │     Agent 7: fixtures + test infrastructure
  ├─► As each completes: merge → dev, verify build
  ├─► When all 6 complete: spawn Agent 8 (integration)
  │
  ▼
PHASE_1_INTEGRATION
  │
  ├─► Spawn Agent 8: integration tests, re-export crate, trybuild
  ├─► Merge → dev
  ├─► Run full Phase 1 acceptance criteria
  ├─► If any fail: spawn fixup agents targeting failures
  ├─► PHASE GATE: all Phase 1 ACs pass
  │
  ▼
PHASE_2_PARALLEL
  │
  ├─► Spawn Agents 9-16 in parallel
  │     Agent 9:  ferrolearn-tree (Decision Tree + Random Forest)
  │     Agent 10: ferrolearn-neighbors (kNN)
  │     Agent 11: ferrolearn-bayes (all NB variants)
  │     Agent 12: ferrolearn-linear additions (Linear SVM)
  │     Agent 13: ferrolearn-cluster (k-Means, DBSCAN)
  │     Agent 14: ferrolearn-decomp (PCA, TruncatedSVD)
  │     Agent 15: ferrolearn-model-sel additions (GridSearchCV, etc.)
  │     Agent 16: ferrolearn-io + ferrolearn-datasets
  ├─► As each completes: merge → dev, verify build
  │
  ▼
PHASE_2_INTEGRATION
  │
  ├─► Spawn Agent 17: Phase 2 fixtures + equivalence docs + integration tests
  ├─► PHASE GATE: all Phase 2 ACs pass
  │
  ▼
PHASE_3_PARALLEL
  │
  ├─► Spawn Agents 18-26 in parallel
  │     Agent 18: Gradient Boosting + HistGB + AdaBoost
  │     Agent 19: GMM, HDBSCAN, Agglomerative Clustering
  │     Agent 20: t-SNE, NMF, Kernel PCA, Kernel SVM
  │     Agent 21: Imputers + Feature Selection
  │     Agent 22: Remaining preprocessors
  │     Agent 23: Backend trait + BLAS + no_std (opus model)
  │     Agent 24: ONNX export + Polars/Arrow
  │     Agent 25: Compile-time pipeline proc macro (opus model)
  │     Agent 26: Statistical equivalence benchmarks
  ├─► PHASE GATE: all Phase 3 ACs pass
  │
  ▼
PHASE_4_PARALLEL
  │
  ├─► Spawn Agents 27-33 in parallel
  │     Agent 27: CudaBackend (opus model)
  │     Agent 28: WgpuBackend (opus model)
  │     Agent 29: PartialFit + SGD + streaming
  │     Agent 30: Calibration + Semi-supervised
  │     Agent 31: ColumnTransformer + UMAP + LDA
  │     Agent 32: Remaining P2 algorithms
  │     Agent 33: Formal verification + benchmarks
  ├─► PHASE GATE: all Phase 4 ACs pass
  │
  ▼
FINALIZE
  │
  ├─► Run full test suite on dev
  ├─► Create PR: dev → main
  ├─► Close all crosslink issues
  ├─► Print summary report
  │
  ▼
DONE
```

### Agent Prompt Template

Each agent receives a prompt built from this template:

```
You are building the `{crate_name}` crate for the ferrolearn project — a scikit-learn
equivalent for Rust.

## Your Assignment
{requirements extracted from phase design doc}

## Acceptance Criteria You Must Satisfy
{acceptance criteria extracted from phase design doc}

## Project Conventions (from CLAUDE.md)
{CLAUDE.md contents}

## Files You Should Create/Modify
{file list from architecture section}

## Dependencies Available
{Cargo.toml dependency block}

## Instructions
1. Create the crate directory and Cargo.toml
2. Implement all requirements listed above
3. Write unit tests for every public function
4. Write doc comments for every public item
5. Run `cargo test -p {crate_name}` and fix any failures
6. Run `cargo clippy -p {crate_name} -- -D warnings` and fix any warnings
7. Commit your work with a descriptive message
8. Report what you delivered and any issues encountered
```

### CLAUDE.md Project Conventions

Written by the coordinator at startup, read by every subagent:

```markdown
# ferrolearn — Project Conventions

## Rust Edition & MSRV
- Edition: 2024
- MSRV: 1.85

## Import Paths
- Core traits: `use ferrolearn_core::{Fit, Predict, Transform, FitTransform}`
- Errors: `use ferrolearn_core::FerroError`
- Dataset: `use ferrolearn_core::Dataset`
- Array types: `use ndarray::{Array1, Array2, ArrayView1, ArrayView2}`
- Float bound: `use num_traits::Float`

## Error Handling
- All public functions return `Result<T, FerroError>`
- Use `thiserror` 2.0 for derive
- Never panic in library code
- Every error variant carries diagnostic context

## Numeric Generics
- Generic bound: `F: Float + Send + Sync + 'static`
- Support both f32 and f64
- Use `num_traits::{Zero, One}` where needed

## Testing Patterns
- Oracle fixtures: load JSON from `fixtures/`, compare with `float_cmp` ULP tolerance
- Property tests: `proptest` with `ProptestConfig::with_cases(256)`
- Fuzz: one target per public fit/transform/predict
- Compile-fail: `trybuild` for type-safety guarantees

## Naming Conventions
- Unfitted: `LinearRegression`, `StandardScaler`
- Fitted: `FittedLinearRegression`, `FittedStandardScaler`
- Traits for introspection: `HasCoefficients`, `HasFeatureImportances`, `HasClasses`

## Crate Dependencies (use these exact versions)
ndarray = "0.17"
faer = "0.24"
sprs = "0.11"
rayon = "1.11"
serde = { version = "1.0", features = ["derive"] }
num-traits = "0.2"
thiserror = "2.0"
```

### Stuck-Agent Protocol

```
DETECT: Agent has been running >15 minutes with no new tool calls
         OR agent reports "I'm blocked" / "I can't figure out"
         OR agent's cargo test fails repeatedly (>3 attempts)

DIAGNOSE:
  1. Read the agent's full output transcript
  2. Identify the category:
     a. COMPILE_ERROR — missing import, wrong type, API mismatch
     b. TEST_FAILURE — logic bug, fixture mismatch, tolerance issue
     c. DESIGN_GAP — the design doc doesn't specify enough detail
     d. DEPENDENCY_CONFLICT — version incompatibility, missing feature flag
     e. SCOPE_CREEP — agent is doing more than assigned

RESPOND:
  a. COMPILE_ERROR → Resume agent with the exact fix (import path, type signature)
  b. TEST_FAILURE → Resume agent with diagnosis and suggested approach
  c. DESIGN_GAP → Coordinator makes the design decision, documents it in crosslink,
     resumes agent with the decision
  d. DEPENDENCY_CONFLICT → Coordinator fixes Cargo.toml on dev branch, tells agent
     to pull latest
  e. SCOPE_CREEP → Resume agent with "Stop. Only implement {X}. Commit what you have."

ESCALATE: If the coordinator cannot resolve after 2 attempts, create a crosslink
  issue tagged `blocker`, describe the problem, and move on to other agents.
  Return to blocked agents after the phase's other work completes.
```

### Context Compression Recovery

The coordinator's persistent state lives in three places:

1. **Crosslink issues** — every agent assignment, status, and outcome
2. **Design docs** — the authoritative task definitions
3. **Git branch state** — `dev` branch shows what's been merged

After context compression, the coordinator runs:

```bash
# What phase are we in?
crosslink issues list --open

# What agents are still running?
# (check background task IDs from recent tool calls)

# What's been merged to dev?
git log --oneline dev

# What tests pass?
cargo test --workspace 2>&1 | tail -20
```

This reconstructs the full state without relying on conversation history.

### Concurrency Limits

- **Max 8 background agents** simultaneously (prevents resource exhaustion on typical hardware)
- **Merges are sequential** — one at a time, with build verification between each
- **Phase gates are synchronization points** — all agents must complete and merge before the next phase starts
- **Fixup agents count toward the 8-agent limit** — if 6 agents are running and 2 fixup agents are needed, the coordinator waits for a slot

### Model Selection Per Agent

| Agent | Model | Rationale |
|-------|-------|-----------|
| 1: ferrolearn-core | opus | Architectural foundation — trait design must be right |
| 2: ferrolearn-sparse | sonnet | Mechanical wrapping of sprs types |
| 3: ferrolearn-metrics | sonnet | Pure math functions, straightforward |
| 4: ferrolearn-preprocess | sonnet | Standard algorithms, well-documented |
| 5: ferrolearn-linear | opus | Custom L-BFGS, numerical sensitivity |
| 6: ferrolearn-model-sel | sonnet | Standard CV patterns |
| 7: fixtures/test infra | sonnet | Scaffolding, not algorithmic |
| 8: integration | opus | Cross-crate interface resolution |
| 9: trees/RF | sonnet | Well-known algorithms |
| 10: kNN | sonnet | Standard spatial indexing |
| 11: Naive Bayes | sonnet | Simple probabilistic models |
| 12: Linear SVM | sonnet | SMO is well-documented |
| 13: clustering | sonnet | Standard algorithms |
| 14: PCA/SVD | sonnet | Delegates to faer |
| 15: GridSearchCV | sonnet | Mechanical parallelism |
| 16: IO/datasets | sonnet | Serialization plumbing |
| 17: Phase 2 integration | opus | Cross-crate resolution |
| 18: Gradient Boosting | opus | HistGB is complex |
| 19: advanced clustering | sonnet | Standard algorithms |
| 20: t-SNE/NMF/kernels | opus | Numerical sensitivity |
| 21: imputers/selection | sonnet | Standard algorithms |
| 22: remaining preprocess | sonnet | Mechanical additions |
| 23: Backend trait | opus | Core abstraction design |
| 24: ONNX/Polars/Arrow | sonnet | Integration plumbing |
| 25: compile-time pipeline | opus | Proc macro + type-level programming |
| 26: stat benchmarks | sonnet | Infrastructure, not algorithms |
| 27: CudaBackend | opus | FFI + GPU memory management |
| 28: WgpuBackend | opus | Compute shaders |
| 29: streaming/SGD | sonnet | Well-documented algorithms |
| 30: calibration/semi-sup | sonnet | Standard sklearn patterns |
| 31: ColumnTransformer/UMAP | opus | UMAP is numerically complex |
| 32: remaining algorithms | sonnet | P2 priority, standard |
| 33: formal verification | opus | Prusti annotations are finicky |
| fixup agents | opus | Debugging requires deep reasoning |

**Breakdown: 13 opus, 20 sonnet, fixups opus**

## Open Questions

*None — all design decisions are resolved in the phase docs.*

## Out of Scope
- The coordinator does not write algorithm code itself — it only orchestrates
- The coordinator does not modify design docs — if a design gap is found, it makes a decision, documents it in crosslink, and instructs the agent
- The coordinator does not handle deployment, CI setup, or publishing to crates.io
- The coordinator does not run the 24-hour fuzz campaign — it sets up fuzz targets but the long run is a human-initiated step
