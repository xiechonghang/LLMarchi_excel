# M1 用户阅读反馈03：纵向结构图与章节层级（2026-10-19）

用户已实际阅读 c97 候选，并明确表示“图已经足够清晰”。记录为当前图清晰度获用户认可，不等于 M1 整体最终阅读通过，也不把工程侧 browser_visual 改为已核验。

新要求只有两项：
1. 模仿 ONNX 图，将结构图改为纵向、主数据流从上到下。保留计算框内源码名和计算类型、箭头上 tensor/权重源码名与 shape、分支/残差/共享/缓存读写条件；不能把整个 SVG 旋转导致文字倾斜，也不能仅把多个横图纵向排列就称完成。
2. 图与图之间用 1、1.1、1.1.1 等章节编号明确包含关系，尤其 Attention → CSA Compressor → Indexer。标题、目录/父子导航、总览对应图的标号一致；不只给扁平图列表顺序编号。Prefill、Decode 各自完整成章，父子关系可追溯，原锚点/联查继续有效。

源码静态依据：hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py，DeepseekV4Attention :794-795 通过 self.compressor 创建所选 CSA compressor；DeepseekV4CSACompressor :618 self.indexer=DeepseekV4Indexer(config)，:690 调用 indexer。用户例示的包含关系符合本样板源码。章节编号只属展示元数据，不改变模型 JSON 或算子粒度。

原已认可算子粒度与 Excel 内容继续保留，本轮仅 HTML 布局/章节导航，不重拆算子、不清冗余、不进入 M2/M3。历史 c97 及更早候选、输入与验收证据只读。新展示须新内容寻址候选，JSON/effective_config/xlsx 与 c97 逐字节一致。完成后需受影响展示复核，再交用户阅读纵向/层级效果。
