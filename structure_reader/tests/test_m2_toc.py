"""分层浏览页的导航契约（原「四变体长页目录」测试随页面替换而迁移）。

旧页面用固定侧边目录 (``.m2-toc``) + ``g-<phase>-<i>`` 锚点；T03 改为整网→层组→
Block→算子的分层路由（``structure.html#/group/<gid>[/block|/ops[/<module>]]``）。
本文件仍检查「导航目标唯一、有界、真实存在」这一原意，只是目标换成层级路由与模块
锚点，不保留旧长页的目录断言。
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent


@unittest.skipUnless((WS / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file(),
                     "no input")
class TestStructurePageNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_network import build_network_overview
        from structure_reader.render.hf_structure_page import render_structure_page
        from structure_reader.extract.hf_layer_structures import build_structures_container

        cls.structures = build_structures_container(WS)
        items = [
            {key: v[key] for key in ("id", "label", "attention_type", "moe_type",
                                     "layer_indices", "representative_layer_index",
                                     "structure")}
            for v in cls.structures["variants"]
        ]
        cls.items = items
        cls.network = build_network_overview(WS)
        cls.html = render_structure_page(cls.network, items).decode("utf-8")

    def test_module_routes_target_existing_details(self):
        h = self.html
        routes = re.findall(r'href="structure\.html#/group/([a-z0-9-]+)/ops/([a-z_]+)"', h)
        self.assertGreater(len(routes), 50)
        for gid, mid in routes:
            self.assertIn(f'id="mod-block-{gid}-{mid}"', h,
                          f"模块路由目标缺失: {gid}/{mid}")
        # 每个模块路由都落在该组自己的 ops 层 section 内
        for gid in {g for g, _ in routes}:
            self.assertIn(f'id="lv-ops-{gid}"', h)

    def test_every_level_link_is_bounded(self):
        h = self.html
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            self.assertIn(f'href="structure.html#/group/{gid}"', h)
            self.assertIn(f'href="structure.html#/group/{gid}/block"', h)
            self.assertIn(f'href="structure.html#/group/{gid}/ops"', h)
        # 没有指向不存在层组的链接
        groups = {v["id"] for v in self.items}
        for target in set(re.findall(r'href="structure\.html#/group/([a-z0-9-]+)"', h)):
            self.assertIn(target, groups)

    def test_router_script_present_and_level_scoped(self):
        h = self.html
        self.assertIn('id="scn-data"', h)
        self.assertIn("hashchange", h)
        self.assertIn("sections.forEach", h)
        self.assertIn('document.body.setAttribute("data-level"', h)
        # 默认层级由 CSS 固定为整网；其余层级默认不显示
        self.assertIn("#lv-network{display:block}", h.replace("\n", ""))
        self.assertIn(".lv{display:none", h.replace("\n", ""))
        # 回到上级的普通链接（不依赖脚本）
        for needle in ("↑ 返回 Block", "↑ 返回整网", "↑ 返回层组"):
            self.assertIn(needle, h)

    def test_no_legacy_long_page_artifacts(self):
        h = self.html
        self.assertNotIn('class="m2-toc"', h)
        self.assertNotIn("decoder_variants.html", h)
        self.assertNotIn("viewer/network.html", h)
        # 同一结构不按 prefill/decode 铺两遍：每层组只有一份算子卡集合
        for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk"):
            self.assertEqual(h.count(f'id="lv-ops-{gid}"'), 1)

    def test_responsive_rules_kept(self):
        h = self.html
        self.assertIn("@media (max-width:1100px)", h)
        self.assertIn("@media (max-width:720px)", h)
        self.assertIn(":focus-visible", h)


if __name__ == "__main__":
    unittest.main()
