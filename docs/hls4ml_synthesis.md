# hls4ml full synthesis — one-time recipe (run elsewhere, NOT on the laptop)

Phase 2 produced an hls4ml project for the trigger encoder
(`models/physics_vae/hls4ml_prj/`, regenerate with `make hls4ml-estimate`)
and a no-synthesis estimate in `reports/phase2/hls4ml_estimate.json`:

- network 57→32→16→8, 2 520 params, 2 464 MACs/event;
- precision `ap_fixed<24,8>` — the tutorial default `ap_fixed<16,6>` is NOT
  usable here (mu sits near a ~1e-3 Sum mu² threshold; 10 fractional bits →
  only 91% p99 trigger-decision agreement; `<24,8>` → 100%, max |Δmu| ≈ 6e-4);
- C-emulation (g++) numerics verified against float PyTorch;
- at ReuseFactor=1 the ~2.5k multipliers exceed a mid-range FPGA's DSP budget —
  RF 4–8 (or 8-bit quantized weights) is the realistic deployment point;
- expected pipeline latency order: tens of cycles at II=RF, i.e. O(100 ns) at
  200 MHz — comfortably inside an L1-style budget.

## Why no local synthesis

Full C-synthesis needs AMD Vitis HLS (~60+ GB install, heavy RAM) — explicitly
out of scope for the 16 GB dev laptop. Everything below is the one-time recipe
for a lab machine / CernVM / Colab-with-Vivado setup.

## Recipe

1. Install Vitis HLS 2023.2+ (or use an LCG/CVMFS machine where it exists).
2. Regenerate the project (or copy `models/physics_vae/hls4ml_prj/` over):

   ```bash
   PYTHONPATH=. python -m scripts.hls4ml_estimate   # writes hls4ml_prj/
   ```

3. Run synthesis from Python (preferred — parses the reports for you):

   ```python
   import hls4ml
   # reload the generated project
   report = hls_model.build(csim=False, synth=True, vsynth=False)
   hls4ml.report.read_vivado_report('models/physics_vae/hls4ml_prj/')
   ```

   or directly: `cd models/physics_vae/hls4ml_prj && vitis_hls -f build_prj.tcl`.

4. Record from the synthesis report into `reports/phase2/hls4ml_estimate.json`
   under a `"synthesis"` key: latency (min/max cycles + ns), II, and the
   DSP/LUT/FF/BRAM utilisation table. Try `ReuseFactor` ∈ {1, 4, 8} and keep
   the smallest RF whose DSP count fits the target part.

## QKeras / QAT status

QKeras is effectively unmaintained for Keras 3-era stacks (our env: TF-free,
PyTorch + hls4ml 1.3) and was not installed. Quantization here is
post-training fixed-point via the hls4ml precision config, which already
achieves exact trigger-decision parity at `<24,8>`. If true
quantization-aware training is wanted later, the maintained path is HGQ2 /
QONNX (Brevitas for PyTorch) — retraining territory, out of Phase 2 scope by
design (frozen Phase 0 checkpoint).
