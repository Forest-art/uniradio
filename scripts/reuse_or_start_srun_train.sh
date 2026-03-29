#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reuse_or_start_srun_train.sh -- "<TRAIN_CMD>"

Behavior:
  1) Reuse an existing running+idle hold allocation if available.
  2) Otherwise wait for a pending hold allocation, or create a new one.
  3) Launch TRAIN_CMD inside that allocation via `srun --jobid ... --overlap`.

Environment defaults:
  ACCOUNT=peilab
  PARTITION=preempt
  QOS=<partition default>
  HOLD_JOB_NAME=dsga-srun-hold
  NODES=1
  GPUS_PER_NODE=1
  CPUS_PER_TASK=8
  MEM=32G
  TIME_LIMIT=24:00:00
  LOG_ROOT=<repo>/results/srun_reuse_logs
  TRAIN_PROCESS_REGEX=torchrun|train_imagenet|train_cifar|accelerate|train_dsga
  ALLOC_WAIT_TIMEOUT_SEC=1800
  ALLOC_POLL_SEC=10
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACCOUNT=${ACCOUNT:-peilab}
PARTITION=${PARTITION:-preempt}
QOS=${QOS:-}
HOLD_JOB_NAME=${HOLD_JOB_NAME:-dsga-srun-hold}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-32G}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
LOG_ROOT=${LOG_ROOT:-${ROOT_DIR}/results/srun_reuse_logs}
RESERVE_ROOT=${RESERVE_ROOT:-${ROOT_DIR}/results/srun_hold_reservations}
TRAIN_PROCESS_REGEX=${TRAIN_PROCESS_REGEX:-torchrun|train_imagenet|train_cifar|accelerate|train_dsga}
ALLOC_WAIT_TIMEOUT_SEC=${ALLOC_WAIT_TIMEOUT_SEC:-1800}
ALLOC_POLL_SEC=${ALLOC_POLL_SEC:-10}

module load slurm >/dev/null 2>&1 || true
export PAGER=${PAGER:-cat}
export SLURM_PAGER=${SLURM_PAGER:-cat}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

TRAIN_CMD="$*"
if [[ -z "${TRAIN_CMD// }" ]]; then
  echo "[error] TRAIN_CMD is empty."
  usage
  exit 1
fi

mkdir -p "${LOG_ROOT}"
mkdir -p "${RESERVE_ROOT}"
SESSION_TS="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="${LOG_ROOT}/${HOLD_JOB_NAME}_${SESSION_TS}"
mkdir -p "${SESSION_DIR}"
RESERVED_JOB=""

log() {
  echo "[$(date +'%F %T')] $*" >&2
}

list_hold_jobs() {
  squeue -h -u "${USER}" -A "${ACCOUNT}" -p "${PARTITION}" -t R,PD \
    -o "%i|%T|%j|%N" \
    | awk -F'|' -v name="${HOLD_JOB_NAME}" '$3==name {print $0}'
}

reservation_dir() {
  local jid="$1"
  echo "${RESERVE_ROOT}/job_${jid}"
}

cleanup_stale_reservation() {
  local jid="$1"
  local rdir
  local owner_pid=""
  rdir="$(reservation_dir "${jid}")"
  [[ -d "${rdir}" ]] || return 0
  if [[ -f "${rdir}/pid" ]]; then
    owner_pid="$(cat "${rdir}/pid" 2>/dev/null || true)"
  fi
  if [[ -n "${owner_pid}" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
    return 1
  fi
  rm -rf "${rdir}"
  return 0
}

reserve_hold_job() {
  local jid="$1"
  local rdir
  rdir="$(reservation_dir "${jid}")"
  if mkdir "${rdir}" 2>/dev/null; then
    echo "$$" > "${rdir}/pid"
    date +%F\ %T > "${rdir}/created_at"
    RESERVED_JOB="${jid}"
    return 0
  fi
  cleanup_stale_reservation "${jid}" || return 1
  if mkdir "${rdir}" 2>/dev/null; then
    echo "$$" > "${rdir}/pid"
    date +%F\ %T > "${rdir}/created_at"
    RESERVED_JOB="${jid}"
    return 0
  fi
  return 1
}

release_hold_job() {
  local rdir
  [[ -n "${RESERVED_JOB}" ]] || return 0
  rdir="$(reservation_dir "${RESERVED_JOB}")"
  rm -rf "${rdir}"
  RESERVED_JOB=""
}

trap release_hold_job EXIT

find_running_idle_hold() {
  local line jid state node busy
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    IFS='|' read -r jid state _ node <<< "${line}"
    [[ "${state}" != "RUNNING" ]] && continue
    busy="$(
      srun --jobid "${jid}" --overlap --ntasks=1 --cpus-per-task=1 \
        bash -lc "pgrep -afu \"${USER}\" -f '${TRAIN_PROCESS_REGEX}' | grep -v 'reuse_or_start_srun_train' | wc -l" \
        2>/dev/null | tr -d '[:space:]'
    )"
    if [[ "${busy}" == "0" ]] && reserve_hold_job "${jid}"; then
      echo "${jid}|${node}"
      return 0
    fi
  done < <(list_hold_jobs)
  return 1
}

find_latest_pending_hold() {
  list_hold_jobs | awk -F'|' '$2=="PENDING" {print $1}' | sort -nr | head -n1
}

wait_until_running() {
  local jid="$1"
  local timeout="$2"
  local poll="$3"
  local elapsed=0
  local state=""
  while (( elapsed <= timeout )); do
    state="$(squeue -j "${jid}" -h -o "%T" || true)"
    if [[ "${state}" == "RUNNING" ]]; then
      return 0
    fi
    if [[ -z "${state}" ]]; then
      return 1
    fi
    sleep "${poll}"
    elapsed=$((elapsed + poll))
  done
  return 1
}

create_new_hold_job() {
  local max_before elapsed jid
  max_before="$(list_hold_jobs | awk -F'|' 'max<$1{max=$1} END {print max+0}')"
  log "No idle hold found. Creating new hold allocation: ${HOLD_JOB_NAME}"
  local -a srun_cmd=(
    srun
    --account "${ACCOUNT}"
    --partition "${PARTITION}"
    --nodes "${NODES}"
    --gpus-per-node "${GPUS_PER_NODE}"
    --cpus-per-task "${CPUS_PER_TASK}"
    --mem "${MEM}"
    --time "${TIME_LIMIT}"
    --job-name "${HOLD_JOB_NAME}"
    --output "${SESSION_DIR}/hold-%j.out"
    --error "${SESSION_DIR}/hold-%j.err"
  )
  if [[ -n "${QOS}" ]]; then
    srun_cmd+=(--qos "${QOS}")
  fi
  srun_cmd+=(
    bash -lc 'echo "[hold] started on $(hostname)"; trap "exit 0" TERM INT; while true; do sleep 300; done'
  )
  nohup "${srun_cmd[@]}" > "${SESSION_DIR}/hold-launch.log" 2>&1 &

  elapsed=0
  while (( elapsed <= ALLOC_WAIT_TIMEOUT_SEC )); do
    jid="$(
      list_hold_jobs | awk -F'|' -v mb="${max_before}" '$1>mb && $2=="RUNNING" {print $1}' \
      | sort -nr | head -n1
    )"
    if [[ -n "${jid:-}" ]]; then
      echo "${jid}"
      return 0
    fi
    sleep "${ALLOC_POLL_SEC}"
    elapsed=$((elapsed + ALLOC_POLL_SEC))
  done
  return 1
}

SELECTED_JOB=""
SELECTED_NODE=""

while [[ -z "${SELECTED_JOB}" ]]; do
  if selected="$(find_running_idle_hold)"; then
    IFS='|' read -r SELECTED_JOB SELECTED_NODE <<< "${selected}"
    log "Reusing idle hold allocation job=${SELECTED_JOB} node=${SELECTED_NODE}"
    break
  fi

  pending_jid="$(find_latest_pending_hold || true)"
  if [[ -n "${pending_jid:-}" ]]; then
    log "Found pending hold allocation job=${pending_jid}, waiting for RUNNING..."
    if wait_until_running "${pending_jid}" "${ALLOC_WAIT_TIMEOUT_SEC}" "${ALLOC_POLL_SEC}"; then
      if reserve_hold_job "${pending_jid}"; then
        SELECTED_JOB="${pending_jid}"
        SELECTED_NODE="$(squeue -j "${SELECTED_JOB}" -h -o "%N" || true)"
        log "Pending hold is now RUNNING and reserved job=${SELECTED_JOB} node=${SELECTED_NODE}"
        break
      fi
      log "Pending hold job=${pending_jid} was reserved by another launcher, retrying."
      sleep "${ALLOC_POLL_SEC}"
      continue
    fi
    log "Pending hold did not become RUNNING in time, creating new hold allocation."
  fi

  new_jid="$(create_new_hold_job)"
  if reserve_hold_job "${new_jid}"; then
    SELECTED_JOB="${new_jid}"
    SELECTED_NODE="$(squeue -j "${SELECTED_JOB}" -h -o "%N" || true)"
    log "Created and reserved hold allocation job=${SELECTED_JOB} node=${SELECTED_NODE}"
    break
  fi
  log "New hold job=${new_jid} was reserved by another launcher, retrying."
  sleep "${ALLOC_POLL_SEC}"
done

if [[ -z "${SELECTED_JOB}" ]]; then
  echo "[error] Failed to obtain a RUNNING hold allocation."
  exit 1
fi

TRAIN_OUT="${SESSION_DIR}/train-${SELECTED_JOB}.out"
TRAIN_ERR="${SESSION_DIR}/train-${SELECTED_JOB}.err"

log "Launching training on allocation job=${SELECTED_JOB}"
log "Session dir: ${SESSION_DIR}"
log "Train stdout: ${TRAIN_OUT}"
log "Train stderr: ${TRAIN_ERR}"

srun --jobid "${SELECTED_JOB}" --overlap \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --output "${TRAIN_OUT}" \
  --error "${TRAIN_ERR}" \
  bash -lc "${TRAIN_CMD}"

log "Training command finished."
