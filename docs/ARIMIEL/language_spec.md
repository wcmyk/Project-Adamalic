# acos language specification (draft)

This draft specification defines the acos language. It is intentionally concrete enough to guide parser and compiler work while leaving room for implementation feedback.

## Design principles
- **Performance with clarity:** Favor zero-cost abstractions and deterministic lifetimes while keeping syntax familiar to Python users.
- **Safety gradient:** Offer opt-in ownership/borrowing (borrow-checked regions) alongside ergonomic defaults and automatic resource release (RAII).
- **Predictable numerics:** Deterministic floating-point policies, explicit precision, and vectorization hints.
- **First-class visualization:** Rendering and shader constructs are part of the core grammar, not a bolted-on DSL.

## Syntax overview
- **Whitespace-aware blocks:** Indentation forms blocks; braces `{}` optional for inline scopes.
- **Type annotations:** `name: Type` on bindings and parameters; type inference fills omissions.
- **Imports:** `use path::{item, *}`; packages resolved via the acos toolchain (compatible with aos modules).
- **Statements/expressions:** Everything is an expression; `let` binds immutable names, `mut` enables mutation.
- **Lambdas:** `|x: T, y| -> U { expr }` or inline `|x| x * x`.
- **Comments:** `//` single-line, `/* ... */` multi-line, and doc comments with `///`.

## Types
- **Primitives:** `bool`, `u8`..`u128`, `i8`..`i128`, `f16`, `f32`, `f64`, `f128`, `char`, `str`.
- **Compound:** tuples `(T, U)`, arrays `[T; N]`, slices `[..T]`, maps `Map[K, V]`, sets `Set[T]`, and NDArray/NDField for tensors.
- **User types:** `struct`, `enum`, `union` (rare, unsafe), and `trait`.
- **Generics:** `struct Box<T: Clone> { value: T }`; variance inferred; where-clauses allowed.
- **Traits/typeclasses:** Ad-hoc polymorphism with blanket impls and specialization gates (`impl<T: Numeric> Add<T> for Array<T>`).
- **Algebraic data types:** Enums with payloads and exhaustive pattern matching.

## Ownership and lifetimes
- **Defaults:** Bindings are immutable and move by default; copy types derive `Copy`.
- **Borrowed references:** `&T` (shared), `&mut T` (unique); lifetimes inferred, with optional named lifetimes `'a` in APIs.
- **Regions:** `borrow { ... }` blocks enforce static checks (optional; omitted for rapid prototyping).
- **RAII:** Destructors (`drop`) run at scope exit; deterministic resource cleanup mirrors C++ semantics.

## Functions and methods
- **Definition:** `fn name(param: Type, param2?: Type = default) -> ReturnType { ... }` (`?` marks optional arg).
- **Methods:** Declared inside `impl Type { ... }`; `self` is explicit and can be `self`, `&self`, or `&mut self`.
- **Overloading:** Function and operator overloading resolved via traits and specialization rules.
- **Inline and constexpr:** `@inline` and `@consteval` attributes mirror C++ hints for performance-sensitive code.

## Control flow
- **Conditionals:** `if/elif/else` as expressions.
- **Loops:** `for`, `while`, `loop` (infinite), with `break`/`continue` carrying values.
- **Pattern matching:**
  ```acos
  match value {
      Point(x, y) if x == y => "diag",
      Vector[..xs] => xs.len(),
      _ => 0,
  }
  ```
- **Exceptions vs Results:** Functions may declare `throws` clauses; otherwise prefer `Result[T, E]` with `?` propagation.

## Modules and packages
- **Files/modules:** Each file is a module; directories form packages. `pub` controls visibility.
- **Re-exporting:** `pub use math::linspace`.
- **Package manager:** `acos pkg add viz` resolves from unified aos/acos registry.

## Concurrency and parallelism
- **Async/await:** `async fn fetch(...) -> Result<Response>`; executors can be thread-pool or evented.
- **Actors:** `actor Logger { inbox: Channel<Log> }` with `receive` handlers.
- **Parallel loops:** `parallel for i, x in array { ... }` emits vectorized/SIMD or GPU kernels when types allow.
- **Dataflow:** `pipeline { stage1 -> stage2 -> stage3 }` for streaming transforms.

## Visualization and shaders
- **Scene graphs:** `scene { layer { rect(...); text(...); } }` compiles to GPU-backed render trees.
- **Shader blocks:**
  ```acos
  shader blur(radius: i32) -> fragment {
      let c = sample(input, uv)
      return avg(kernel(radius, uv), c)
  }
  ```
- **Render targets:** `figure().grid(2,2).add(surface(shader=blur))`.

## Compilation model
1. **Parse:** aos-compatible grammar with acos extensions.
2. **Lowering:** AST → typed HIR with borrow/lifetime annotations.
3. **IR:** SSA-based mid-level IR with effect types and purity markers.
4. **Optimization:** Inlining, LICM, auto-vectorization hints, and memory-layout specialization (SoA/AoS toggles).
5. **Codegen:** LLVM for CPU, SPIR-V/Metal for GPU, or transpile-to-C++ for bootstrap compilers.
6. **Artifacts:** Static binaries, shared libs, and aos-compatible modules; metadata includes capability flags.

## Tooling expectations
- **acos CLI:** `acos new`, `acos run`, `acos test`, `acos fmt`, `acos doc`.
- **LSP/formatter:** Standardized formatting rules (PEP8-style whitespace, C++-style trailing return types optional), semantic tokens for editors.
- **Testing:** `test` blocks inside modules compile to test binaries; property testing via `proptest` crate.

## Safety and interop
- **Unsafe blocks:** Required for manual pointer arithmetic, union access, or FFI beyond safe wrappers.
- **FFI:** `extern "C" fn` definitions, `c_struct` layout attribute, and zero-copy views over numpy/pyarrow buffers.
- **Determinism:** Reproducible floating-point modes (`strict`, `fast-math`), and stable parallel reduction options.

## Standard library sketch
- `core`: primitives, traits (`Eq`, `Ord`, `Hash`, `Display`, `Clone`, `Copy`), iterators, options/results.
- `collections`: vector, deque, hashmap, btree, priority queue.
- `async`: futures, channels, timers, actor runtime.
- `math`: linear algebra, random, stats, special functions.
- `tensor`: NDArray/NDField, autograd, mixed precision, JIT fusion.
- `viz`: plotting, scene graphs, GPU pipelines, SVG/PDF rasterization.
- `ffi`: C/C++ bridge, numpy/pyarrow interop.

## Compatibility with aos
- **Parsing:** aos code is accepted unchanged.
- **ABI:** Exported symbols default to aos layouts; acos-only features mark modules with capability metadata so aos loaders can degrade gracefully.
- **Migration aids:** Compiler flags `--emit-aos` (transpile) and `--check-aos-compat` (lint) help incremental adoption.

## Error model
- **Diagnostics:** Colorized, span-rich compiler errors with suggestions and fix-its.
- **Contracts:** `requires/ensures` design-by-contract; compiled to runtime checks or proofs in `debug`/`release` modes.
- **Logging:** Built-in structured logging macros (`info!`, `trace!`, `error!`) that are no-ops in minimal builds.

## Open questions (for iteration)
- Borrow checker design space (Rust-like vs region-based inference) for the optional `borrow` blocks.
- Specialization rules to keep trait resolution deterministic.
- Balancing whitespace significance with inline blocks when mixing aos legacy code.
- Determining the default parallel runtime for `parallel for` (threaded vs GPU heuristics).
