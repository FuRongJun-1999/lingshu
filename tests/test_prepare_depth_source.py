"""The native preparation patch is explicit, idempotent and fails closed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


_PATH = Path(__file__).resolve().parents[1] / "tools/reconstruct/prepare_depth_source.py"
_SPEC = importlib.util.spec_from_file_location("prepare_depth_source", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class PrepareDepthSourceTests(unittest.TestCase):
    def test_patch_has_expected_content_preserves_eol_and_is_idempotent(self):
        anchor = (b"    } else {\n"
                  b"        std::array<float,12> ext; std::array<float,9> intr;\n"
                  b"        if (!c->engine->depth_pose_native(img, depth, second, ext, intr, H, W)){")
        prefix = (b"    } else if (!out_ext && !out_intr){\n"
                  b"        if (!c->engine->depth_native_image(img, depth, second, H, W) || second.size() != depth.size()){\n"
                  b'            c->last_error = "depth_dense: depth-only graph failed"; return -1; }\n')
        for eol in (b"\n", b"\r\n"):
            with self.subTest(eol=eol), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / "src/da_capi.cpp"
                path.parent.mkdir()
                original = b"// before\n" + anchor + b"\n// after\n"
                path.write_bytes(original.replace(b"\n", eol))
                self.assertTrue(_MODULE.prepare_source(root))
                expected = (b"// before\n" + prefix + anchor + b"\n// after\n").replace(b"\n", eol)
                self.assertEqual(path.read_bytes(), expected)
                self.assertFalse(_MODULE.prepare_source(root))
                self.assertEqual(path.read_bytes(), expected)

    def test_unexpected_or_ambiguous_source_is_not_written(self):
        for original in (b"// no expected anchor\n", _MODULE._ANCHOR * 2,
                         _MODULE._MARKER + b"\n" + _MODULE._ANCHOR):
            with self.subTest(source=original[:40]), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "da_capi.cpp"
                path.write_bytes(original)
                with self.assertRaises(ValueError):
                    _MODULE.patch_depth_source(path)
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
