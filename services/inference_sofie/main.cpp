// PHAROS Phase 2 -- standalone SOFIE trigger-score binary (BLAS only, no ROOT).
//
// Compile only after generated/encoder_mu.hxx exists (see README.md; the
// header is emitted by scripts/sofie_probe.py inside a SOFIE-enabled ROOT).
//
// Protocol: one event per line on stdin, 57 whitespace-separated floats
// (already normalized -- the Python scorer owns the log1p/z-score transform).
// Prints Sum mu^2 (sum of squared latent means) per line on stdout, one
// result per input line, flushed per line so a driving process can pipeline.

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "generated/encoder_mu.hxx"

constexpr int kNFeatures = 57;

int main() {
    // Weights (.dat) sit next to the generated header.
    TMVA_SOFIE_encoder_mu::Session session("generated/encoder_mu.dat");

    std::string line;
    std::vector<float> x(kNFeatures);
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        for (int i = 0; i < kNFeatures; ++i) {
            if (!(iss >> x[i])) {
                std::fprintf(stderr, "bad input line (need %d floats)\n",
                             kNFeatures);
                return 1;
            }
        }
        std::vector<float> mu = session.infer(x.data());
        double score = 0.0;
        for (float m : mu) score += static_cast<double>(m) * m;
        std::printf("%.9g\n", score);
        std::fflush(stdout);
    }
    return 0;
}
