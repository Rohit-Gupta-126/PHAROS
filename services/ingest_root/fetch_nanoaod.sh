#!/usr/bin/env bash
# Fetch one CMS Open Data NanoAOD file into data/raw/cms_opendata/ over HTTPS.
#
# Runs inside a one-shot rootproject/root container (curl + python3 present):
#   docker run --rm -v "$PWD:/work" -w /work rootproject/root:latest \
#       bash services/ingest_root/fetch_nanoaod.sh
#
# Default source is the CMS Open Data ZeroBias PFNano file (record 31316,
# /ZeroBias/Run2016G-UL2016_MiniAODv2_PFNanoAODv1). ZeroBias = random
# bunch-crossing readout with no physics trigger, i.e. an inclusive/unbiased
# minimum-bias sample -- the right basis for a sim-to-real domain-gap study. It
# carries the full standard NanoAOD content (Electron/Muon/Jet/MET) plus PF
# candidates. Override the source with CMS_NANOAOD_URL. The file is ~1.05 GB.
#
# Why HTTPS + curl -k (not xrdcp/root://):
#   - rootproject/root:latest no longer bundles an XRootD client (no xrdcp CLI,
#     no libXrdCl.so.3), so neither xrdcp nor RDataFrame root:// streaming works.
#     We download over HTTPS to a local file and ingest that.
#   - eospublic.cern.ch redirects to an EOS gateway whose TLS chain is the CERN
#     Grid CA, which is not in the container's default CA bundle. Rather than
#     fatten the image with the CERN CA, we use `curl -k` and instead verify
#     integrity out-of-band with the record's published adler32 checksum below.
#     Trust comes from the checksum match, not the TLS chain.
set -euo pipefail

DEST_DIR="${CMS_DEST_DIR:-data/raw/cms_opendata}"
URL="${CMS_NANOAOD_URL:-https://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/ZeroBias/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240212_182529/0000/nano_data2016_1-1.root}"
# Published adler32 for the default file (opendata record 31316 file index).
# Only enforced when fetching the default URL; override CMS_ADLER32 (empty to
# skip) when pointing CMS_NANOAOD_URL at a different file.
EXPECT_ADLER32="${CMS_ADLER32:-9a8ac2aa}"
DEST="${DEST_DIR}/$(basename "${URL}")"

mkdir -p "${DEST_DIR}"
if [[ -f "${DEST}" ]]; then
  echo "[fetch-cms] already present: ${DEST}"
else
  echo "[fetch-cms] curl -k ${URL}"
  echo "[fetch-cms]   -> ${DEST}"
  # -k: accept the CERN Grid CA (integrity is checked via adler32 below).
  # -L: follow the eospublic -> EOS gateway redirect. -f: fail on HTTP errors.
  curl -kLf --retry 3 -o "${DEST}" "${URL}"
fi

echo "[fetch-cms] size: $(du -h "${DEST}" | cut -f1) ($(stat -c%s "${DEST}") bytes) ${DEST}"

if [[ -n "${EXPECT_ADLER32}" ]]; then
  echo "[fetch-cms] verifying adler32 (expect ${EXPECT_ADLER32}) ..."
  GOT_ADLER32="$(python3 - "${DEST}" <<'PY'
import sys, zlib
a = 1
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        a = zlib.adler32(chunk, a)
print(format(a & 0xffffffff, "08x"))
PY
)"
  echo "[fetch-cms] adler32: ${GOT_ADLER32}"
  if [[ "${GOT_ADLER32}" != "${EXPECT_ADLER32}" ]]; then
    echo "[fetch-cms] ERROR: adler32 mismatch (got ${GOT_ADLER32}, expected ${EXPECT_ADLER32})" >&2
    echo "[fetch-cms] download is corrupt/incomplete; removing ${DEST}" >&2
    rm -f "${DEST}"
    exit 1
  fi
  echo "[fetch-cms] adler32 OK"
fi

echo "[fetch-cms] done: ${DEST}"
