# 数值不可用的预测验证

UnifiedWorldModel.verify 与 core.world_model("verify") 仅对目标时刻匹配、三轴坐标有限、
bound 有限且非负、距离可表示的项计分。无效项保留 pending，不进入命中率分母。

unverifiable_reason 区分 invalid_observation、invalid_prediction、invalid_bound 与
unrepresentable_distance。验证返回里的坏坐标/bound 为 None，原缓存与历史读数保持原样。
有限统计的求和溢出采用缩放均值；正常路径、缺失/陈旧观测闸与原阈值保留。

pending 不代表预测命中或失败。该接口不修复坏轨迹、不改变原观测接收规则，
对验证返回的严格 JSON 检查不等于原始 perceive/history 全部可严格编码。
[固定提交复跑、变异及未测边界](testing/world-verify-finite-evidence-20261028.md)。
