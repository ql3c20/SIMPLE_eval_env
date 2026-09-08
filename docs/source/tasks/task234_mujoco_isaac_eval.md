# Task2/3/4 MuJoCo + Isaac eval

严格复现入口：

| Task | Gym ID |
|---|---|
| 2 | `simple/G1Fullstate20260805Task2IsaacEval-v0` |
| 3 | `simple/G1Fullstate20260804Task3IsaacEval-v0` |
| 4 | `simple/G1Fullstate20260729Task4IsaacEval-v0` |

启动 eval 前先检查录制快照、初始状态、模型维度/关节语义和 HSSD：

```bash
PYTHONPATH=src uv run --no-sync \
  python scripts/task234_reproduction_preflight.py --task all \
  --manifest-dir outputs/task234_manifests
```

录制资产不在默认路径时，只覆盖路径，不复制或修改源文件：

```bash
export SIMPLE_TASK2_RECORDING_ROOT=/path/to/20260805_task2_new
export SIMPLE_TASK3_RECORDING_ROOT=/path/to/20260804_task3_new
export SIMPLE_TASK4_RECORDING_ROOT=/path/to/20260729_task4
```

也可以分别用 `SIMPLE_TASK{2,3,4}_RECORDING_INSTANCE` 选择 episode。
Isaac 派生 USD 默认缓存到
`/tmp/simple_isaac_mjcf_cache/<mjcf-bundle-sha256>/scene.usd`，可用
`SIMPLE_ISAAC_MJCF_CACHE` 修改缓存根目录。

DDS 到 UDP 的只保留最新状态 relay：

```bash
python scripts/task234_mujoco_dds_udp_relay.py \
  --dds-python-path /path/to/directory/containing/dds_types.py \
  --host 127.0.0.1 --port 23331 --rate 60
```

relay 使用 `BestEffort + KeepLast(1)`；UDP 包包含完整的
`nq/nv/qpos/qvel/sim_time/running`。Isaac 中的任务 USD 和 HSSD 均禁用碰撞，
且严格任务不会推进 Isaac 的正常物理时间。策略输入只允许
`head_stereo_left` 的 `640x480 uint8 RGB`。
