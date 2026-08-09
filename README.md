# inference_simulation

Validation and benchmarking harness for the exported anomaly-detection ONNX
models. It is the Python counterpart of the C++ production engine
(`ONNX_inference/AnomalyEngine.cpp`) and is kept **byte-for-byte contract-driven**
with it: every decision (thresholds, display bounds, colour order) is read from
the calibration metadata embedded in the `.onnx`, so a model that behaves
correctly here behaves identically in production.

## What it does

For every image in a folder it:

1. Runs the ONNX model (FP32 / FP16 / INT8 via TensorRT, or CUDA / CPU).
2. Classifies it: raw graph `anomaly_score` `>= image_threshold_raw` → **ANOMALY**,
   exactly the comparison the C++ engine makes.
3. Renders a heatmap overlay: the raw `anomaly_map` min-max normalized with the
   embedded `map_min_raw`/`map_max_raw` bounds, colour-mapped and alpha-blended,
   with defect contours drawn at `pixel_threshold_raw` (same as the C++ engine).

Then it benchmarks raw inference throughput at **batch size 1 and 17** (default)
at the requested precision. Timing wraps `session.run` only (no pre/post-processing).

## Alignment with the C++ engine

| Aspect | Behaviour (shared with `AnomalyEngine.cpp`) |
|---|---|
| Image verdict | raw `anomaly_score` `>= image_threshold_raw` |
| Heatmap normalization | min-max with `map_min_raw` / `map_max_raw` |
| Pixel mask / contours | raw map `>= pixel_threshold_raw` |
| Input normalization | inside the graph (`normalize_inside_graph=true`, host feeds float32 `[0,1]`) |
| Colour order | BGR→RGB per `preproc_color_conversion` |
| Host-side blur / dynamic crop | none (removed) |

**Known discrepancy to reconcile on the C++ side:** the models declare
`preproc_resize_interpolation=bilinear` with `preproc_antialias=true` (this is how
the thresholds were calibrated). Python follows the contract and resizes with
PIL's antialiased bilinear filter; the C++ engine currently downsamples with
`cv2.INTER_AREA`, which contradicts the metadata and should be aligned.

## Execution provider (TensorRT → CUDA → CPU)

The default device is **`tensorrt`** (classic `TensorrtExecutionProvider`),
configured like the C++ engine (`OrtsessionConfig.cpp`): device 0, FP16/INT8 per
`--precision`, engine + timing caches (namespaced per precision under
`--engine_cache_dir`), the most aggressive builder search
(`trt_builder_optimization_level=5`), and an explicit dynamic-batch profile
(`trt_profile_min/opt/max_shapes`) so one engine covers the whole requested batch
range without a per-shape rebuild.

The provider list is `[TensorRT, CUDA, CPU]`, so ONNX Runtime applies its native
**TensorRT → CUDA → CPU** fallback *within a single session*: a subgraph TensorRT
cannot build/run drops to CUDA, then CPU — mirroring the C++ cascade. The
optimization (`opt`) profile shape defaults to the smallest requested batch
(`--trt_opt_batch` to override) to keep the engine build feasible on large models.

Select a different backend explicitly with `--device cuda` or `--device cpu`.

## OOM behaviour and the batch-size / VRAM trade-off

`gpu_mem_limit` is unset by default (ORT uses the available VRAM); cap it manually
with `--gpu_mem_limit <bytes>` if needed. If a batch size still exhausts memory it
is reported as `oom` in the results and skipped, so the other batch sizes still
run. INT8 is refused for models declaring `quantization_safe=false` (PatchCore)
before any engine is built.

### Why VRAM usage grows with batch size (and why PatchCore only runs at small batches)

Peak VRAM during inference is dominated by the **activations / intermediate
tensors**, and those scale **linearly with the batch size**: doubling the batch
roughly doubles the memory held for every intermediate tensor produced along the
network. Model *weights* are a fixed cost paid once; the *activations* are paid
per image in the batch. So a model that fits comfortably at batch 1 can exceed the
GPU at batch 8 or 17 purely because the intermediates are now 8× or 17× larger.

This is especially severe for **PatchCore**. Its verdict comes from a
nearest-neighbour search of every image patch against a large **memory bank**
(`inner.memory_bank = [97484, 192]`, ~97k reference patches). The distance
computation materializes intermediate tensors sized roughly
`batch × num_patches × memory_bank_size`, so its peak VRAM grows *much* faster
with the batch than a plain CNN's does. On a 6 GB RTX 4050 this measured out as:

| Batch | PatchCore |
|---|---|
| 1 | OK (~134 ms) |
| 2 | OK (~343 ms) |
| 4 | starts spilling (~11 s) |
| 8+ | out of memory |

The other architectures (SuperSimpleNet, SK-RD4AD, EfficientAD) are ordinary
feed-forward CNNs whose activations grow linearly and moderately, so they scale to
larger batches on the same GPU.

**Thesis takeaway:** PatchCore's memory-bank nearest-neighbour design makes its
inference memory grow super-linearly with the workload and caps the usable batch
size on commodity GPUs. This is a concrete, quantified motivation for moving to
architectures whose cost is a fixed forward pass (segmentation/reconstruction or
distillation heads: SuperSimpleNet, SK-RD4AD, EfficientAD), which keep VRAM
bounded and batch size free to trade off against throughput.

## Usage

```bash
python main.py \
  --model models/<model>.onnx \
  --input_dir <folder of images> \
  --output_dir output/<model> \
  --device tensorrt        # default (TensorRT→CUDA→CPU); or cuda / cpu
  --precision fp32         # or fp16 / int8 (tensorrt only)
  --batch_sizes 1,17       # default
```

Key flags: `--threshold` (override the embedded `image_threshold_raw`),
`--precision` (fp32/fp16/int8), `--colormap`, `--overlay_alpha`, `--warmup_iters`,
`--timed_iters`, `--gpu_mem_limit`, `--trt_workspace_gb`, `--trt_opt_batch`,
`--calibration_table` (INT8).

Outputs land in `--output_dir`: `heatmaps/`, `benchmark_results.txt`,
`benchmark_results.json`, `per_image_results.csv`.
