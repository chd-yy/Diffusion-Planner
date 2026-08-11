#!/usr/bin/env bash
# 仅在 264 个 shard 全部完成后冻结 100k manifest 并做全局复验；不会启动训练。

set -euo pipefail

project_root="${HDP_PROJECT_ROOT:-/root/autodl-tmp/workspace/Diffusion-Planner}"
python_bin="${HDP_PYTHON_BIN:-/root/autodl-tmp/conda_envs/diffusion_planner/bin/python}"
plan_path="${HDP_PLAN_PATH:-/root/autodl-tmp/nuplan/plans/nuplan_train_100k/preprocessing_plan.json}"
output_root="${HDP_OUTPUT_ROOT:-/root/autodl-tmp/processed/nuplan_train_100k_gate}"
manifest_path="$output_root/train_manifest.json"
merged_report="$output_root/train_merged_report.json"
validation_report="$output_root/train_cache_validation_report.json"

required_paths=(
  "$python_bin"
  "$plan_path"
  "$output_root"
  "$project_root/HDP-nuplan/scripts/merge_preprocessing_shards.py"
  "$project_root/HDP-nuplan/scripts/validate_processed_cache.py"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "缺少最终校验所需路径：$required_path" >&2
    exit 1
  fi
done

if pgrep -f '[r]un_preprocessing_range.py|[d]ownload_nuplan_log_subset.py|[r]un_preprocessing_shard.py|/data_process.py' >/dev/null; then
  echo "仍有下载或预处理进程，拒绝在数据变化期间冻结 manifest。" >&2
  exit 1
fi

if pgrep -f '[t]rain_predictor.py|[t]rain_predictor_rl.py|[t]orch.distributed.run' >/dev/null; then
  echo "检测到训练进程，拒绝并发执行最终数据校验。" >&2
  exit 1
fi

"$python_bin" "$project_root/HDP-nuplan/scripts/merge_preprocessing_shards.py" \
  --shards_root "$output_root" \
  --plan "$plan_path" \
  --output_manifest "$manifest_path" \
  --output_report "$merged_report"

selected_log_count="$(
  "$python_bin" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_log_count"])' \
    "$merged_report"
)"

"$python_bin" "$project_root/HDP-nuplan/scripts/validate_processed_cache.py" \
  --cache_dir "$output_root" \
  --manifest "$manifest_path" \
  --sampling_report "$merged_report" \
  --expected_count 100000 \
  --expected_log_count "$selected_log_count" \
  --output "$validation_report"

echo "最终数据冻结与全局复验完成："
echo "  manifest=$manifest_path"
echo "  merged_report=$merged_report"
echo "  validation_report=$validation_report"
echo "本脚本到此结束；没有启动监督训练或 RL 训练。"
