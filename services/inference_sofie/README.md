# services/inference_sofie — SOFIE C++ trigger inference (future work)

Status (Phase 2): **deferred — ONNX Runtime is the runnable path.**

The probe (`reports/phase2/sofie_probe.txt`) showed `rootproject/root:latest`
ships the SOFIE runtime but not the ONNX parser, and building ROOT with
`-Dtmva-sofie=ON` is too heavy for the dev laptop. The recipe lives in
`docker/root-sofie.Dockerfile` — run it once on a bigger machine, then:

```bash
docker run --rm -v "$PWD:/work" -w /work pharos/root-sofie \
    python3 scripts/sofie_probe.py        # emits generated/encoder_mu.hxx + .dat
bash services/inference_sofie/build.sh    # compiles sofie_score (BLAS only, no ROOT)
```

`main.cpp` is written against SOFIE's standard generated interface
(`TMVA_SOFIE_encoder_mu::Session`) and is ready to compile as soon as
`generated/encoder_mu.hxx` exists. The binary reads one 57-float
whitespace-separated event per line on stdin and prints `Sum mu^2` per line —
the same protocol `physics_scorer_sofie --backend sofie` speaks.

Until then, use the documented fallback:

```bash
python -m services.scorers.physics_scorer_sofie --backend ort
```
