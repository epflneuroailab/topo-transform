#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
S3_ROOT_URL="s3://epfl-neuroailab-public/david/topo-transform"
S3_URL="s3://epfl-neuroailab-public/david/topo-transform/debug.zip"
HTTPS_URL="https://epfl-neuroailab-public.s3.amazonaws.com/david/topo-transform/debug.zip"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache"
TARGET_DIR="$CACHE_DIR/debug"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download cached data for topo-transform.

Default behavior:
  Download debug.zip, extract it into cache/debug, and remove debug.zip.
  If 7z/7zz/7za is available, it is used for multithreaded zip extraction.

Options:
  --full
      Recursively download everything under:
        $S3_ROOT_URL
      into:
        $CACHE_DIR
      Archives are not decompressed.

  -h, --help
      Show this help message.
EOF
}

FULL_DOWNLOAD=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --full)
            FULL_DOWNLOAD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "$CACHE_DIR"

if [ "$FULL_DOWNLOAD" -eq 1 ]; then
    if ! command -v aws >/dev/null 2>&1; then
        echo "Error: aws cli is required for recursive S3 downloads from $S3_ROOT_URL" >&2
        exit 1
    fi

    echo "Recursively downloading $S3_ROOT_URL into $CACHE_DIR..."
    aws s3 cp "$S3_ROOT_URL" "$CACHE_DIR" --recursive
    echo "Done. Downloaded into $CACHE_DIR/"
    exit 0
fi

cd "$CACHE_DIR"

FILENAME="$(basename "$S3_URL")"

# # --- Download ---
# echo "Downloading $FILENAME..."
# if command -v aws >/dev/null 2>&1; then
#     echo "Using aws cli from $S3_URL"
#     aws s3 cp "$S3_URL" "$FILENAME"
# else
#     echo "Using HTTPS fallback from $HTTPS_URL"
#     curl -fL -o "$FILENAME" "$HTTPS_URL"
# fi

# --- Extract ---
echo "Extracting $FILENAME..."
TMP_EXTRACT_DIR="$(mktemp -d "$CACHE_DIR/.extract.XXXXXX")"
cleanup_tmp() {
    rm -rf "$TMP_EXTRACT_DIR"
}
trap cleanup_tmp EXIT

if command -v 7zz >/dev/null 2>&1; then
    7zz x -bd -bso0 -bsp1 -mmt=on -y "-o$TMP_EXTRACT_DIR" "$FILENAME"
elif command -v 7z >/dev/null 2>&1; then
    7z x -bd -bso0 -bsp1 -mmt=on -y "-o$TMP_EXTRACT_DIR" "$FILENAME"
elif command -v 7za >/dev/null 2>&1; then
    7za x -bd -bso0 -bsp1 -mmt=on -y "-o$TMP_EXTRACT_DIR" "$FILENAME"
elif command -v unzip >/dev/null 2>&1; then
    unzip -oq "$FILENAME" -d "$TMP_EXTRACT_DIR"
elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$FILENAME" -C "$TMP_EXTRACT_DIR"
else
    echo "Error: none of 7zz, 7z, 7za, unzip, or bsdtar is available to extract $FILENAME" >&2
    exit 1
fi

# Normalize output so extracted contents always live under cache/debug/**.
SOURCE_DIR="$TMP_EXTRACT_DIR"
if [ -d "$TMP_EXTRACT_DIR/debug" ]; then
    SOURCE_DIR="$TMP_EXTRACT_DIR/debug"
fi
rm -rf "$TARGET_DIR"
mv "$SOURCE_DIR" "$TARGET_DIR"
trap - EXIT
cleanup_tmp

# --- Cleanup ---
echo "Removing $FILENAME..."
rm -f "$FILENAME"

echo "✅ Done. Extracted into $TARGET_DIR/"
