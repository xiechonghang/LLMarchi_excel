# layout-v2-rework独立复核：工程通过，待用户布局确认

日期：2026-11-03。依据第二轮布局反馈与layout-v2-codex-review，定点复核P1/P2与最终版本绑定。新版xlsx：`953592e594cdccfa7aed8914f5fed791cc10a0e972e6bc585ee4b5c3be751699`；累积patch：`9f49f41ffb5a60fb09b2076e7f0ad8cc954281a0aff081e5347a1315d1d203d6`。

## Spec轴：通过

- 主表通过参数层V_*校验值取维度；场景值检查数值、非布尔、整数和非负，head/group/width等基础配置要求正整数。派生、普通与复合路径共用校验后的解析入口；H_bias_on与Nhit_ok也使用校验值。
- Codex独立用Artifact Tool导入副本重算q_a_proj：seqlen=-1、空白、1.5、TRUE、数字文本8、普通文本abc均使T/cube/op为未知；seqlen=0仍为合法0，seqlen=8保持8192。结果见本目录 `E01-layout-v2-rework-codex-verify/input-cases.json`。
- Codex独立执行交付verify_workbook_v2.py，用bundled Python/Node与既有LibreOffice Kit实际重算，passed=true、recalc=true、errors=[]。覆盖非法维度与吞吐、固定batch、合法成本、派生联动、条件0、复合repeat及旧导出。独立回执见 `E01-layout-v2-rework-codex-verify/verification.json`。
- 291实例=282计费+8excluded+1container，101模板，2979次使用；mHC7×86、QKV8×43保持，合法N_hit=6下两场景CUBE/VEC/op总成本与前轮一致。
- 9项layout-v2测试由Codex独立运行通过；主表维度保持短公式，没有恢复逐行长守卫。原始参数隐藏表为默认来源，用户可在可见基础参数格填写覆盖值，后续V_*仍校验。

## 交付与显示轴：通过

- 两次独立冻结输入重建都得到最终xlsx相同哈希；patch仅4个Excel相关新增文件，与workspace源码一致，apply-check通过。
- manifest中交付哈希、验证、patch与重算副本一致；旧版xlsx/patch保留，保护清单11项匹配。
- 9张截图哈希匹配，底层渲染来源绑定重算副本 `1dffb7c4640240a7dc33a2bcf0d763e7ce6a03ddd24985ec0dd1614327a27619`。
- 两张区块截图已实际查看；A列宽28、标题高28，21区块标题与次数完整可见，Hash/TopK专家切片清楚区分，无重复横幅。
- 复现说明的最终哈希和根目录切换已修正。未发现新增工程阻塞。

## 停点

新版独立样表工程通过，可交用户确认布局。用户体验尚未确认，E03正式集成未启动，patch未应用原源码；HTML及历史候选保持。原生Microsoft Excel未实际验证，LibreOffice/Artifact Tool重算与截图不代签其字体、打印与原生行为。不自动关Issue、不提交推送、不恢复自动化。
