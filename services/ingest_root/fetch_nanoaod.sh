#!/usr/bin/env bash
# Fetch one CMS Open Data NanoAOD file into data/raw/cms_opendata/.
#
# Runs inside a one-shot rootproject/root container (xrdcp + XRootD are bundled):
#   docker run --rm -v "$PWD:/work" -w /work rootproject/root:latest \
#       bash services/ingest_root/fetch_nanoaod.sh
#
# Default is a standard Run2016 UL NanoAODv9 file (all Electron/Muon/Jet/MET
# collections present). Override the source with CMS_NANOAOD_URL. The file is
# ~hundreds of MB; if the download is too heavy or the endpoint is unreachable,
# STOP and pick a smaller record rather than grinding (project stop-rule) --
# ingest_nanoaod.py can also stream the root:// URL directly with no local copy.
set -euo pipefail

DEST_DIR="${CMS_DEST_DIR:-data/raw/cms_opendata}"
URL="${CMS_NANOAOD_URL:-root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/127C2975-1B1C-A046-AABF-62B77E757A86.root}"
DEST="${DEST_DIR}/$(basename "${URL}")"

mkdir -p "${DEST_DIR}"
if [[ -f "${DEST}" ]]; then
  echo "[fetch-cms] already present: ${DEST}"
  exit 0
fi

echo "[fetch-cms] xrdcp ${URL}"
echo "[fetch-cms]   -> ${DEST}"
xrdcp --force "${URL}" "${DEST}"
echo "[fetch-cms] done: $(du -h "${DEST}" | cut -f1) ${DEST}"
