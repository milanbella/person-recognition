set -e

LOCAL_DIR="/c/wi/luxonis/person-recognition/"
REMOTE_USER="ovie1"
REMOTE_HOST="ovie"
REMOTE_DIR="/home/ovie1/person-recognition/"

rsync -avz --delete \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='src/.venv' \
    --exclude='src/.cache' \
    --exclude='src/.depthai_cached_models/' \
    --exclude='src/recordings/' \
    --exclude='src/plane_calibrations/' \
    --exclude='src/evidence/' \
    --exclude='src/embedding_runs/' \
    --exclude='src/identity_runs/' \
    --exclude='src/entry_session_runs/' \
    "$LOCAL_DIR" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "Sync completed."
