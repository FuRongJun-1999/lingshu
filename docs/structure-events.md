# 结构事件与版本历史写入（提案）

对应 issue #329。此件采用公共事件入口和共同留史实现；版本记录使用独立的便利方法，外部锚点的原授权闸保持。类型/授权取舍供项目维护者裁定，不表示待裁事项已获通过。

## 使用

外部来源在受信宿主中调用已有 D-007 授权的入口：

```python
node = engine.record_structure_event(
    "important_event", "完成一次外部校准",
    designer_key=host_designer_key,
    source="校准来源说明", metadata={"weight": 0.8})

version = engine.record_version_event(
    "0.2", "增加版本迭代记录入口",
    designer_key=host_designer_key, from_version="0.1",
    status="planned", source="发布流程")
```

host_designer_key 由宿主持有，不能发送给不可信插件。source 和 metadata 是来源描述，不能作为授权依据。辅助信息可以缺省，不添加 URL、内容摘要或额外验证清单。

版本状态：recorded（默认）、planned、applied、failed、rolled_back、uncertain。方法只保存调用者提供的记录，不执行升级，不把 release note 或 planned 证明为 applied。执行器应按其实际结果调用；这种调用仍须受信宿主授权。

原 register_external_anchor 继续只接受 pre_access_stance、introspection、external_calibration；版本类型不混入外部锚点目录。该方法增加可选 source/metadata，原位置参数保持兼容。

## 持久记录与角色语义

共同实现用于外部锚点、重要事件、版本迭代，以及已有注意力调整、坐标迁移和验证标准留史。正文保持 [kind] 内容，类型在 tags 和 state_attributes 中显式保存：

```json
{
  "structure_event": {
    "kind": "version_iteration",
    "source": "发布流程",
    "record_class": "history",
    "metadata": {
      "version": "0.2",
      "from_version": "0.1",
      "status": "planned",
      "details": {}
    }
  }
}
```

PRIMARY 写 STRUCTURE；其他角色写 KNOWLEDGE 并统一带 pending_sync。每次调用追加新事件，不经知识层 M5 正文去重，相同文字的不同发生不会被吞。pending_sync 仅表达待同步，**不承诺已经到达父节点**，#165 的实际同步仍未实现。

公开历史入口默认 confidence=0.8，长期保存不等于独立验证为真。既有外部锚点及内部已执行动作保留原置信度以兼容现有调用。record_class=history 不授予规则生效或执行权限。

标签、来源和正文一次插入，操作记录与节点同一事务提交；复用 nodes/action_logs，不新增表、索引、依赖或备份格式。现有 get_node、get_nodes_by_tag、M13 备份可以读回这些信息。

## 失败与事务

- 坐标迁移的数据变更、事件节点和操作记录一起提交；留史失败回滚迁移，异常向调用方传播。
- 没有待迁移坐标时不取得迁移写锁。
- 调用方已有事务时，事件工作单元只管理自己的 savepoint，不提前提交调用方工作；最终持久化由调用方 commit 决定。
- 注意力调整和验证标准仍由原路径先执行；留史失败改为明确抛异常，不再静默返回成功。它们涉及内存配置和原有终裁提交，**本件没有实现跨内存/数据库整体回滚**；异常不能被理解为原动作未执行。

notify_event/consume_events 仍是易失唤醒机制。需要保留的重要事件应先调用持久入口，队列防抖不作为事实记录或去重机制。

## 安全与范围

本件沿用现有 D-007 和角色边界，提供受信宿主可调用的持久化原语。私有方法只是内部 API 约定，**不能隔离同进程敌对代码**。

部署必须将写服务、数据库、密钥与审批权放在不可信 worker 的进程和权限边界之外；由受限 IPC 映射真实身份，只暴露公共业务命令，不允许任意 SQL/对象调用或角色自报。本件没有提供 OS 沙箱、连接认证或隔离部署，不宣称关闭 #156，也不修复 #125/#127/#140 的其他旧写入通道。

本件没有改写晋升审批政策、引入哈希/审计链、建设通用证据检查框架、增加消息中间件或复制脑端代码。它完成 #329 的写入口及重复留史收口提案；完整治理和隔离需后续独立接入。

## 本地回归

```powershell
python -X utf8 -m pytest tests/test_structure_events.py tests/test_issue109_external_anchor_gate.py tests/test_core_legacy_schema_migration.py -q --disable-warnings
```

新增仅覆盖常用业务：版本记录/授权/备份、SUB 多次事件、迁移留史故障回滚、调用方事务、已有配置路径。没有添加哈希、数值边缘或组合爆炸测试。现有 CI 自动收集该 pytest 文件。
