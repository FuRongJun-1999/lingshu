# 结构历史与受保护写服务

对应 #329。提供外部锚点、重要事件和版本迭代的持久写入，并区分历史记录和已批准的现行结构。轻核仅使用 Python 标准库。

## 接入方式

受信宿主可继续使用既有 D-007 密钥入口：

```python
engine.record_structure_event(
    "important_event", "完成外部校准", designer_key=host_designer_key,
    source="校准来源说明", metadata={"weight": 0.8})
engine.record_version_event(
    "0.2", "增加版本记录", designer_key=host_designer_key,
    from_version="0.1", status="planned")
```

不可信插件/agent 使用 `StructureClient`，在独立的 OS 账户下调用单独运行的写服务。它们没有数据库、服务配置、设计者密钥或其他角色的客户端私钥。服务直接装配 `LayeredStore`，不加载引擎的可选插件。

```python
from lingshu.core.structure_service import StructureClient

client = StructureClient(
    "127.0.0.1", 7443, ca="service-ca.pem",
    certificate="worker.pem", key="worker.key", server_name="localhost")
event = client.call(
    "record_event", source_event_id="calibration-42", kind="external_calibration",
    payload={"text": "校准观察", "weight": 0.8}, metadata={"note": "人工采集"})
proposal = client.call(
    "propose", request_id="proposal-42", aggregate_key="attention.reference",
    kind="rule", payload={"weight": 0.8, "scope": "本实例"},
    expected_revision=0, event_id=event["id"])
```

复核端从 `get_proposal` 读取服务保存的内容，然后 `review(..., decision="approve")`；批准端使用同一个 `proposal_id` 调用 `approve`。执行方通过 `current(aggregate_key=...)` 读取现行批准，再评估其业务范围；JSON 规则不会被写服务作为代码执行。

## 授权与部署

客户端和服务端使用双向 TLS。客户端证书必须来自专用受信 CA，并包含一个 `URI:urn:lingshu:<身份>` 的 SAN。服务将已验证证书映射到自己的权限表；请求不能自报 subject、namespace、role、capabilities 或数据库路径。服务证书须包含实际主机的 DNS/IP SAN；客户端验证证书链和主机名。

私钥由部署者在仓库之外生成；CA 签发私钥保留在独立管理端，服务只需要 CA 公共证书。不得给 worker 签发复核/批准身份，或将这些私钥共享给同一个不可信进程。TLS 接口依据 [Python SSL](https://docs.python.org/3/library/ssl.html) 的证书认证机制实现。

配置文件放在专用受保护目录。所有路径相对于此目录，并必须留在目录内：

```json
{
  "database": "body.sqlite3",
  "certificate": "server.pem",
  "key": "server.key",
  "ca": "client-ca.pem",
  "listen": ["127.0.0.1", 7443],
  "principals": [
    {
      "identity": "urn:lingshu:worker", "subject": "worker",
      "namespace": "body", "source": "worker",
      "capabilities": ["events.write", "structure.propose", "structure.read"]
    },
    {
      "identity": "urn:lingshu:reviewer", "subject": "reviewer",
      "namespace": "body", "source": "reviewer",
      "capabilities": ["structure.read", "structure.review"]
    },
    {
      "identity": "urn:lingshu:approver", "subject": "approver",
      "namespace": "body", "source": "approver",
      "capabilities": ["structure.read", "structure.approve"]
    },
    {
      "identity": "urn:lingshu:executor", "subject": "executor",
      "namespace": "body", "source": "executor",
      "capabilities": ["structure.read", "versions.report"],
      "instances": ["body-1"]
    }
  ]
}
```

- POSIX：服务目录由专用服务账户拥有，权限 `0700`；worker 使用不同的非特权 UID。可用系统服务管理器运行，工作目录与 worker 工作区分开。
- Windows：服务与 worker 使用不同的普通账户。专用目录及已有文件的 DACL 仅允许服务账户、SYSTEM 和 Administrators；所有者也须属于这些受信身份。运行前由管理员在此专用目录设置 ACL；程序不修改用户目录或创建系统账户。启动检查采用 [GetNamedSecurityInfoW](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-getnamedsecurityinfow)。

部署目录的父级路径及服务程序也由服务账户或系统管理员控制，不能位于 worker 可替换、重命名或修改代码的目录中。目录内的数据库、配置和服务私钥由管理端创建，不复用 worker 创建的文件或硬链接。启动权限检查针对配置目录及受保护文件，不替部署者验证整台主机的账户、父目录和代码安装权限。

用服务账户执行：

```text
python -m lingshu.core.structure_service service-state/service.json
```

启动时检查目录/已有受保护文件权限、配置路径和数据库版本；不在每次读取时复查全部历史或文件摘要。最多同时处理 16 个连接，认证/传输有超时，单条 JSON 消息上限 1 MiB；历史查询使用分页。不接受 SQL、pickle、代码、动态导入或任意对象方法。

本接口不创建 OS 账户，也不代替 worker 的 OS 沙箱：同账户恶意代码或管理员仍能直接访问该账户资源。部署隔离是必要条件，不是给函数加私有前缀就获得的保证。修改证书权限映射后重启服务；无须因此冻结已有批准。

## 记录与治理

| 命令 | 权限 | 持久行为 |
|---|---|---|
| record_event | events.write | 追加来源声明，origin=reported；没有规范执行权 |
| record_version_result | versions.report + 指定实例 | 记录授权执行器上报的已有结果，origin=executor_report |
| propose | structure.propose | 固定目标、正文、范围/期限、目标版本及来源事件 |
| review | structure.review | approve / needs_info / reject；提案人不能自审 |
| approve | structure.approve | 复核已通过且三方身份互斥；按目标版本提交 |
| get_proposal / current / target_status / event_history / structure_history | structure.read | 仅查询认证身份的 namespace |

提案正文不会原地修改。修改内容使用新的 request_id，并可关联 prior_proposal_id；原提案人可以替代自己的未关闭提案。被拒绝的尝试保留 rejected。批准只针对已保存 proposal_id，修改客户端读回副本不会改变批准对象。

批准时检查 expected_revision。并发变更返回 Conflict，既有批准不被覆盖；使用当前版本重新提案。撤销也追加版本；撤销/到期后 `current` 返回空，但 `target_status` 保留 revision，避免把曾存在的对象当作全新目标。规则/锚点的 payload 应明确包含适用范围；写服务负责批准与存储，具体业务条件由受信运行时评估。

历史记录不经过正文相似度去重。受信宿主直接入口每次追加；服务接口按 namespace/source/source_event_id 识别重试，相同 ID、相同 kind/origin/payload 返回原记录。辅助 metadata 或时间说明变化不覆盖原值；正文变化须追加纠正事件。不同来源事件 ID 即使正文相同也各自保留。

普通事件类型和辅助来源 URL 不加额外门槛；不明确的发生时间保留原说明并标注待澄清，不阻断事件记录。规范有效期会影响行为，须明确时区。公开历史置信度为 0.8；长期保存及 executor_report 都不等于独立证明内容为真。

## 版本、事务与恢复

宿主版本入口支持 recorded、planned、applied、failed、rolled_back、uncertain；这些是宿主提供的历史声明。服务的执行结果入口要求 report_id、run_id、instance_id、to_version 和 outcome；outcome 只接受 applied/failed/rolled_back/uncertain，并校验执行器绑定实例。版本计划用普通事件记录。成功写入返回 id、origin、recorded_at、replayed 和 warnings 的简短回执；正文用历史查询读取。收到版本结果不执行升级；中断后追加同一 run_id 的 uncertain/对账记录，不覆盖旧结果。

事件正文、元数据、节点投影和操作记录同一事务提交。治理扩展复用原 SQLite 库与 action_logs，并新增五张 structure_* 表；批准、目标头、节点和裁决同时提交。已有普通数据库无需初始化治理；旧有角色写入兼容：PRIMARY 写 STRUCTURE，其他角色写 KNOWLEDGE + pending_sync。pending_sync 的实际父节点交付仍属 #165 的独立同步功能。

迁移数据与迁移结果一起提交；留史失败会回滚。注意力权重及验证标准的持久配置、裁决和历史同事务；失败恢复权重/不发布新配置，重启从 engine_meta 恢复已提交值。注入的注意力策略 set_weight 必须只改内存，不能执行外部 I/O。配置动作要求自有事务；普通事件在调用方事务里只使用 savepoint，不提前提交调用方。

`get_current_structure_nodes` 与 self_check 排除普通历史及失效规范；旧格式结构保留读取语义。完整历史仍可通过 store 或服务分页查询。直接引擎写入口只供受信宿主；worker 只拿客户端，不获得原引擎对象。

M13 节点 JSON 可备份事件投影，但不是治理全库备份。使用受信管理命令保存 SQLite 的一致快照：

```text
python -m lingshu.core.structure_service service-state/service.json --backup service-state/backup.sqlite3
```

该命令不占用服务监听端口，可在服务运行时备份；恢复时停止服务，以完整快照恢复数据库及已部署权限，再启动。治理库检测到只有 schema 标记却缺治理表会拒绝启动，避免把不完整 JSON 恢复误认作完整治理。备份路径不能是正在运行的数据库。

## 回归

新增回归围绕配置故障回滚、提案快照/复核/批准/撤销/版本冲突、全库备份恢复和真实 mTLS 子进程调用；测试证书只在临时目录生成。认证流程不使用伪造 ActorContext 当作隔离证明。Windows ACL 已通过本地链路；POSIX 权限及两档 Python 由 CI 运行。没有引入内容摘要链、审计哈希或数值边缘测试。
