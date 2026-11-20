# 实例内视觉引用缓冲只读查询

`engine.list_local_visual_frames(entity_id=None, limit=50)` 查询当前实例由 `ingest_frame()` 留下的引用元数据。缓冲仅在当前引擎内存中存在，最多保留最新200条，不是持久化帧库。

## 查询口径

- 输出是新建字典列表，每项恰有 `entity_id`、`frame_ref`、`ts`；不生成frame id、尺寸、source或hash。
- 按摄入追加顺序从新到旧返回；不会按时间戳重新排序，因此系统时钟回退或同时间戳不改变先后顺序。
- `entity_id=None` 表示不过滤；其他值须为非空字符串，按精确ID匹配，不解析实体名称。
- `limit` 是0..200的整数，拒绝bool；默认50，0返回空列表；先按实体过滤再取数量。
- `frame_ref` 只允许字符串或None作为不透明引用，查询不open/stat/resolve/normalize。内部选中记录若带bytes或其他对象，明确拒绝，不回传潜在原始像素。
- 时间戳须为有限数值，实体标识须为字符串或None；不默默跳过坏记录。

## 生命周期与副作用

该接口不访问SQLite连接、不执行SQL或提交，不修改buffer；关闭引擎后显式抛出包含“已关闭”的 `sqlite3.ProgrammingError`，不把关闭后的缓存当可用实例。

`prepare_shared_sync()` 仍只报告 `local_buffer_count`，引用与原始视觉不进入共享载荷。返回引用可能包含调用方自己的路径，调用方负责后续展示权限；此接口不读取引用内容。
