# Visualization and ML stack in acos

acos is designed to outclass MATLAB for numerics and deliver richer visuals than typical ML pipelines. This document describes the visual/ML architecture and how it stays aligned with the language core.

## Goals
- **Publication-quality by default:** High-DPI, color-managed outputs across raster, vector, and interactive targets.
- **GPU-first:** Unified CPU/GPU tensor backend with automatic kernel fusion and scheduling.
- **Declarative + programmable:** Authors can compose charts declaratively while dropping to shaders or kernels when needed.
- **Interop-ready:** Interchange tensors and renderables with Python (numpy, torch), C++ (Eigen, xtensor), and aos ecosystems.

## Tensor and numeric stack
- **NDArray/NDField:** Unified tensor types with shape/dtype tracked at compile time when available; dynamic fallback at runtime.
- **Autograd:** Reverse-mode with static graph capture or eager tape; fused kernels lower to LLVM or SPIR-V.
- **Mixed precision:** `f16`, `bf16`, `tf32`, and `f64` with automatic loss-scaling and error estimates.
- **Vectorized control flow:** `vif/vwhile` constructs allow per-element branching to stay SIMD/GPU friendly.
- **Sparse + structured:** CSR/COO, block-sparse, and low-rank primitives with auto-tuning.

## Visualization pipeline
1. **Scene graph:** `viz::scene` builds a retained-mode graph: layers, shapes, text, shaders, and data bindings.
2. **Layout engine:** Constraint-based layout with optional auto-gridding and responsive sizing.
3. **Styling:** Style sheets (CSS-like) that support themes and accessibility presets (colorblind-safe palettes, WCAG contrast).
4. **Rendering:**
   - **GPU path:** tessellation → batching → shader compilation (SPIR-V/Metal) → render targets.
   - **CPU path:** high-quality path rendering + font rasterization (HarfBuzz, FreeType equivalents).
5. **Export:** PNG, SVG, PDF, WebGPU canvas, mp4/webm animations, and live widgets for notebooks.

## Charting DSL
```acos
use viz::{figure, surface, palette}

fn main() {
    let xs = linspace(-3.14, 3.14, 256)
    let ys = xs
    let zs = xs.outer(ys, |x, y| sin(x) * cos(y))

    figure()
        .title("Sine-Cosine Surface")
        .theme(palette::viridis())
        .add(surface(xs, ys, zs).shaded(mode="pbr").wireframe(alpha=0.25))
        .export("surface.svg")
}
```

## ML integration
- **Model definition:** `module` blocks declare parameter collections; layers are composable functions with shape/type constraints.
- **Training loop:** `optimizer` traits (SGD, Adam, Lion) and `schedule` combinators; `grad tape` spans blocks.
- **Inference runtime:** Ahead-of-time compilation to CPU/GPU with quantization passes (int8/int4, per-channel scales).
- **Deployment:** Emits static libraries with C ABI and a minimal runtime; optional wasm for browser delivery.

## Better-than-MATLAB visuals
- High-DPI default (2x), adaptive tick/label placement, publication themes, LaTeX-quality math text, and perceptually uniform palettes.
- Built-in animated charts (`.animate(step=...)`) and shader-backed primitives for volumetric rendering.
- `viz::notebook` integrates with Jupyter and web dashboards without Python glue code.

## Better-than-typical-ML visual workflow
- **Observability hooks:** Metrics + traces piped directly into interactive dashboards rendered by the viz stack.
- **Model introspection:** Activation and gradient heatmaps rendered via GPU pipelines; `viz::explain` for attribution maps.
- **Data provenance:** Every figure embeds metadata (data hashes, commit IDs) for reproducibility.

## Migration and interop
- **From aos:** Import aos visualization bindings; acos layers add shader blocks and tensor-backed datasets transparently.
- **From Python:** `py import` bridge exposes numpy/torch tensors as zero-copy views; matplotlib plots can be imported and restyled using the acos scene graph.
- **From C++:** Header-only bridge exposes NDArray and scene graph builders; shaders compiled ahead-of-time to reuse existing GPU pipelines.
