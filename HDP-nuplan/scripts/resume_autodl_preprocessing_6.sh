#!/usr/bin/env bash
# AutoDL 开机后安全恢复 NuPlan 100k 的 6 路滚动预处理；本脚本不会启动训练。

set -euo pipefail

project_root="${HDP_PROJECT_ROOT:-/root/autodl-tmp/workspace/Diffusion-Planner}"
python_bin="${HDP_PYTHON_BIN:-/root/autodl-tmp/conda_envs/diffusion_planner/bin/python}"
plan_path="${HDP_PLAN_PATH:-/root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json}"
archive_index="${HDP_ARCHIVE_INDEX:-/root/autodl-tmp/nuplan/archive_index_train_v1.1.json}"
raw_root="${HDP_RAW_ROOT:-/root/autodl-tmp/nuplan/raw}"
map_path="${HDP_MAP_PATH:-/root/autodl-tmp/nuplan/maps}"
output_root="${HDP_OUTPUT_ROOT:-/root/autodl-tmp/processed/nuplan_train_100k_gate}"
log_root="${HDP_LOG_ROOT:-/root/autodl-tmp/logs}"
worker_count=6

required_paths=(
  "$python_bin"
  "$project_root/HDP-nuplan/scripts/run_preprocessing_range.py"
  "$project_root/HDP-nuplan/scripts/download_nuplan_log_subset.py"
  "$plan_path"
  "$archive_index"
  "$map_path"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "缺少恢复所需路径：$required_path" >&2
    exit 1
  fi
done

if pgrep -f '[r]un_preprocessing_range.py|[d]ownload_nuplan_log_subset.py|[r]un_preprocessing_shard.py|/data_process.py' >/dev/null; then
  echo "检测到已有下载或预处理进程，拒绝重复启动。" >&2
  exit 1
fi

if pgrep -f '[t]rain_predictor.py|[t]rain_predictor_rl.py|[t]orch.distributed.run' >/dev/null; then
  echo "检测到训练进程；按当前任务约束拒绝恢复预处理。" >&2
  exit 1
fi

mkdir -p "$raw_root" "$output_root" "$log_root"
free_kib="$(df -Pk "$raw_root" | awk 'NR == 2 {print $4}')"
minimum_kib=$((20 * 1024 * 1024))
if (( free_kib < minimum_kib )); then
  echo "数据盘可用空间不足 20 GiB，拒绝启动。" >&2
  exit 1
fi

echo "恢复前定向测试：下载断点、临时文件清理和 worker 互斥分配"
"$python_bin" -m pytest -q \
  "$project_root/HDP-nuplan/tests/test_nuplan_subset_download.py" \
  "$project_root/HDP-nuplan/tests/test_preprocessing_range.py"

echo "残留 member 临时文件数量：$(find "$raw_root" -type f -name '.*.tmp' | wc -l)"
echo "完整 DB 会校验后跳过；死亡 PID 的同名临时文件会在对应 DB 重试前清理。"

for worker_index in $(seq 0 $((worker_count - 1))); do
  state_path="$log_root/rolling_worker_${worker_index}_of_6_state.json"
  if [[ ! -f "$state_path" ]]; then
    echo "缺少 worker 断点状态：$state_path" >&2
    exit 1
  fi
done

declare -a launched_pids=()
for worker_index in $(seq 0 $((worker_count - 1))); do
  state_path="$log_root/rolling_worker_${worker_index}_of_6_state.json"
  worker_log="$log_root/rolling_worker_${worker_index}_of_6.log"
  nohup /usr/bin/time -v env PYTHONUNBUFFERED=1 \
    "$python_bin" \
    "$project_root/HDP-nuplan/scripts/run_preprocessing_range.py" \
      --plan "$plan_path" \
      --archive_index "$archive_index" \
      --raw_root "$raw_root" \
      --map_path "$map_path" \
      --output_root "$output_root" \
      --state_path "$state_path" \
      --start_index 0 \
      --end_index 264 \
      --worker_index "$worker_index" \
      --worker_count "$worker_count" \
      --checksum_mode files \
      --cleanup_raw \
      --resume \
      --connect_timeout_seconds 10 \
      --read_timeout_seconds 60 \
      --max_member_retries 3 \
      --retry_delay_seconds 5 \
      --download_backend remotezip \
      --min_free_gib 20 \
      >>"$worker_log" 2>&1 &
  launched_pids+=("$!")
  echo "worker $worker_index 已启动：outer_pid=$!，state=$state_path"
done

sleep 3
failed=0
for index in "${!launched_pids[@]}"; do
  if ! kill -0 "${launched_pids[$index]}" 2>/dev/null; then
    echo "worker $index 启动后提前退出，请检查对应日志。" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "至少一个 worker 未保持运行；先排查，禁止继续追加 worker。" >&2
  exit 1
fi

echo "6 路预处理已从原 state 恢复；未启动监督训练或 RL 训练。"
