# acos — the accelerated aos extension

acos (pronounced *ah-kos*) is a next-generation programming language built as a forward-compatible extension of **aos**. It fuses **C++-grade performance** with **Python-grade readability** to deliver a productive language for scientific computing, simulation, and visualization that can outperform MATLAB for numerics and modern ML stacks for visual output.

## Why acos
- **Speed with safety:** Ahead-of-time compilation, cache-friendly layouts, deterministic RAII, and optional borrow-checked lifetimes keep performance near hand-tuned C++.
- **Readable by design:** Whitespace-aware blocks, expressive literals, and minimal punctuation keep code approachable for Python users.
- **Visual-first standard library:** Built-in plotting, GPU-accelerated tensor ops, and shader authoring pipelines target publication-quality visuals with minimal boilerplate.
- **ML-aware numerics:** First-class NDArray/NDField types with autograd, mixed precision, and vectorized control flow allow prototyping and deployment in one language.
- **Interoperable:** Direct FFI with C/C++, zero-copy views over numpy/pyarrow buffers, and a unified package manager for aos/acos modules.

## Relationship to aos
- **Superset syntax:** Every valid aos program parses under acos. New constructs (traits, pattern matching, async, ownership modifiers) are opt-in.
- **Shared ABI:** acos objects are layout-compatible with aos where possible; module metadata advertises any extended features.
- **Gradual migration:** The compiler can emit aos-compatible artifacts or fully optimized acos-native binaries.

## Feature highlights
- **Types & inference:** Structural and nominal types, generics, typeclasses/traits, and flow-sensitive type inference.
- **Memory model:** Deterministic destruction (RAII) with move semantics, copy elision, and opt-in borrow-checked regions for safety-critical code.
- **Concurrency:** `async/await`, actor mailboxes, and data-parallel kernels (`parallel for`, `map`) that target CPU SIMD or GPU backends.
- **Pattern matching:** Exhaustive `match` with guards, destructuring, and ergonomics for algebraic data types.
- **Error handling:** Expressions return `Result[T, E]` or raise typed exceptions; interop surfaces both.
- **Visualization stack:** `plot`/`viz` modules, declarative scene graphs, and shader kernels embedded via multiline blocks.

## Quick example
```acos
use math::{linspace, sin}
use viz::{figure, line}

fn damped_osc(freq: f64, decay: f64) -> (Array<f64>, Array<f64>) {
    let t = linspace(0.0, 10.0, 2000)
    let y = sin(2.0 * PI * freq * t) * exp(-decay * t)
    return (t, y)
}

trait Solver {
    fn solve(self, t: Array<f64>) -> Array<f64>
}

struct RK4 { step: f64 }

impl Solver for RK4 {
    fn solve(self, t) {
        parallel for i, ti in t {
            // borrow-checked capture of buffers
            y[i] = step_rk4(ti)
        }
        return y
    }
}

fn main() {
    let (t, y) = damped_osc(freq=2.5, decay=0.2)
    figure().title("Damped oscillator").add(line(t, y)).save("osc.png")
}
```

## Repository contents
- `language_spec.md` — detailed syntax, semantics, and compilation model.
- `visualization_and_ml.md` — visual pipeline, tensor stack, and GPU strategy.
- `examples/` — idiomatic code sketches for numerics, visualization, and systems code.

## Roadmap (high level)
1. **Compiler front-end:** aos-compatible parser with acos extensions, SSA IR, and type inference.
2. **Backends:** CPU (LLVM), GPU (SPIR-V/Metal), and a fast transpile-to-C++ path for early adopters.
3. **Standard library:** `core` (types, traits), `math`, `tensor`, `viz`, `async`, and `ffi` modules.
4. **Tooling:** `acos` CLI (build/run/test), package manager, LSP server, formatter, and doc generator.
5. **Interop bridges:** numpy/pyarrow zero-copy views, C/C++ headers, and aos module compatibility shims.
