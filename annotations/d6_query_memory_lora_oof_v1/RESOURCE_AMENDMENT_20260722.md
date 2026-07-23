# D6 GPU 资源执行修订

日期：2026-07-22

状态：用户在 trainability smoke 因外部进程抢占而暂停后明确授权。

本修订只改变 GPU 调度，不改变 D6 冻结候选、数据、fold、训练超参数、
early stopping、决策头或评价门。原始协议和 JSON config 保持字节不变，现有
session-boundary checkpoint 继续绑定原 config SHA256。

用户授权 D6 与无关 compute process 共享物理 GPU，不再要求启动时无外部进程，
也不再要求启动时 free memory 至少 75 GiB。共享模式使用显式 CLI flag，启动时
仍要求至少 24 GiB free memory。该值高于已观测 trainability 进程占用约 10.2 GiB
的两倍，并保留显著余量；运行后的 PyTorch peak allocated `<=70 GiB` 和 session
timeout 门保持不变。若发生 CUDA OOM、非有限张量或资源门失败，实验停止，不
调整模型结构或训练配置。

所有共享执行必须在 environment、contract、launcher status 和最终报告中披露。
外部进程不得被终止、迁移或修改。
