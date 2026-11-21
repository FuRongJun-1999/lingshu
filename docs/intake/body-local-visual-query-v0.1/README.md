# 身体实例视觉引用查询 v0.1 · 新件登记

来源：本仓新增只读公开方法，复用现有 `ingest_frame()` 的200条实例内缓存。

- 仅返回真实已存字段entity_id/frame_ref/ts，按追加顺序倒序；过滤使用精确entity_id，limit 0..200。
- 字符串/None引用不解析、不打开、不规范化；拒绝bytes及任意对象引用，不回传像素。返回新建dict，不修改缓存。
- 不执行SQL、不写文件/数据库/记忆、不改变共享同步数据；关闭后fail-closed。
- 缓冲是实例内元数据，不是持久化图像、视频或语义能力，不添加entity来源或虚构frame id。

验证面：真实摄入201+条裁剪、实体过滤、时钟回退、limit与非法参数、缓存隔离、路径探测禁用、SQLite trace零SQL、total_changes零变化、同步计数及关闭错误。

## 复跑

```sh
python -m pytest tests/test_core_local_visual_buffer.py tests/test_batch122_entry_type_gates_guard.py tests/test_issue660_close_slot_guard.py tests/test_issue165_shared_sync_payload.py tests/test_issue463_shared_sync_identity_guard.py tests/test_manifest_integrity.py tests/test_issue156_component_discovery.py tests/test_doc_module_counts_guard.py -q --no-header
# 128 passed
python -X utf8 tools/verify_manifest.py --json --strict
# exit_code=0
```

已更新core.py对应LF归一源指纹；py_compile与git diff --check通过。
