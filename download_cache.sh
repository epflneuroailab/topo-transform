#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
S3_URL="s3://epfl-neuroailab-public/david/topo-transform/debug.zip"
HTTPS_URL="https://epfl-neuroailab-public.s3.amazonaws.com/david/topo-transform/debug.zip"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache"
TARGET_DIR="$CACHE_DIR/debug"
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

FILENAME="$(basename "$S3_URL")"

# --- Download ---
echo "Downloading $FILENAME..."
if command -v aws >/dev/null 2>&1; then
    echo "Using aws cli from $S3_URL"
    aws s3 cp "$S3_URL" "$FILENAME" --no-progress
else
    echo "Using HTTPS fallback from $HTTPS_URL"
    curl -fL -o "$FILENAME" "$HTTPS_URL"
fi

# --- Extract ---
echo "Extracting $FILENAME..."
TMP_EXTRACT_DIR="$(mktemp -d "$CACHE_DIR/.extract.XXXXXX")"
if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$FILENAME" -d "$TMP_EXTRACT_DIR"
elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$FILENAME" -C "$TMP_EXTRACT_DIR"
else
    echo "Error: neither unzip nor bsdtar is available to extract $FILENAME" >&2
    exit 1
fi

# Normalize output so extracted contents always live under cache/debug/**.
SOURCE_DIR="$TMP_EXTRACT_DIR"
if [ -d "$TMP_EXTRACT_DIR/debug" ]; then
    SOURCE_DIR="$TMP_EXTRACT_DIR/debug"
fi
mkdir -p "$TARGET_DIR"
cp -a "$SOURCE_DIR"/. "$TARGET_DIR"/
rm -rf "$TMP_EXTRACT_DIR"

# --- Cleanup ---
echo "Removing $FILENAME..."
rm -f "$FILENAME"

echo "✅ Done. Extracted into $TARGET_DIR/"
