# High-Speed Policy v0.2.0

## 發布狀態

`model_1298.pt` 已由完成的 High-Speed v0.2.0 training run 提升為目前的
project-local deployment candidate。肉眼播放確認沒有異常抖動，速度表現
正常；播放時繞圈是同時出現前進、側移與 yaw command 的預期結果。

正式固定速度結果見
[High-Speed v0.2.0 固定速度測試](high_speed_evaluation.md)。此版本通過
各軸 command 邊界測試，但三軸同時最大仍有 tracking trade-off，因此不是
實體機的安全速度宣告。

## 正式產物

Checkpoint：

`checkpoints/anymal_d_locomotion_v1/high_speed_v0.2.0/model_1298.pt`

匯出目錄：

`exported/anymal_d_locomotion_v1/high_speed_v0.2.0/`

| 產物 | SHA256 |
|---|---|
| `model_1298.pt` | `a25799ceb15090d1a733ce49d77b444c9bedd325b7fd36d095b2b85e94f2c55d` |
| `policy.pt` | `869ad3ccf60375dd9cff2e3f2abe16493a234310e9c6da22f1ab626de9697dc6` |
| `policy.onnx` | `e83a150970838dad16f4015618f799eb1f1bcb9ad174a70fa9caa80cac06cea0` |

同一目錄也包含：

- `policy_metadata.yaml`：policy contract、joint order、版本與 artifact hash
- `parity_validation.json`：checkpoint／TorchScript／ONNX 數值比較報告

這些 binary 與 generated metadata 保留在 project root 內，依 `.gitignore`
不提交至 GitHub。Git 追蹤本文件、契約與驗證工具。

## Output parity

驗證使用 seed 42 產生 256 組 deterministic float32 observation，包含全零、
固定線性範圍及 seeded random inputs。輸入與輸出維度分別為 48 與 12，
absolute／relative tolerance 均為 `1e-5`。

| 比較 | 最大絕對誤差 | 平均絕對誤差 | 結果 |
|---|---:|---:|---|
| checkpoint vs TorchScript | `0.0` | `0.0` | Pass |
| checkpoint vs ONNX | `3.8146973e-6` | `4.5433322e-7` | Pass |
| TorchScript vs ONNX | `3.8146973e-6` | `4.5433322e-7` | Pass |

Checkpoint 路徑直接重建 ELU actor `48 → 128 → 128 → 128 → 12`；TorchScript
由 `torch.jit.load` 執行；ONNX 先通過 `onnx.checker`，再由
`onnx.reference.ReferenceEvaluator` 執行。

重跑指令：

```bash
cd /home/ros/anymal_locomotion

TERM=xterm-256color PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p \
  scripts/validation/validate_policy_parity.py \
  --checkpoint checkpoints/anymal_d_locomotion_v1/high_speed_v0.2.0/model_1298.pt \
  --jit exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy.pt \
  --onnx exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy.onnx \
  --agent-config \
    logs/rsl_rl/anymal_d_locomotion_v1/2026-07-23_21-24-11_high_speed_v020_from_baseline_1000iter/params/agent.yaml \
  --samples 256 --seed 42 \
  --output \
    exported/anymal_d_locomotion_v1/high_speed_v0.2.0/parity_validation.json
```

目前 Isaac Lab Python 環境沒有 ONNX Runtime，因此本輪證明的是 checkpoint、
TorchScript 與 ONNX graph 的數值一致性。未來選定 ROS 2 deployment 的實際
ONNX backend（例如 ONNX Runtime 或 TensorRT）後，仍須以同一組 inputs
增加 backend-specific parity test。
