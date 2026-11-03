import unittest
import onnx
from onnx import TensorProto, helper

from structure_reader.onnx_module.build import apply_display_dims, display_dim, set_static_dims


class TestOnnxDimNames(unittest.TestCase):
    def setUp(self):
        set_static_dims({})
    def _model(self):
        dims = ["B", "T", "H", "Rq", "B×T", "min(P+T,W-1)", "2Dh", 512, "unk__0"]
        vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, dims)
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["Dh", "unk__0"])
        node = helper.make_node("Identity", ["x"], ["y"], name="n")
        g = helper.make_graph([node], "g", [vi], [out])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
        m.ir_version = 10
        return m

    def test_mapping_known_ops_unknown_and_idempotent(self):
        m = self._model()
        apply_display_dims(m)
        in_dims = [d.dim_param or d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        out_dims = [d.dim_param or d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
        self.assertEqual(in_dims, ["batch_size", "sequence_length", "hidden_size",
                                   "q_lora_rank", "batch_size×sequence_length",
                                   "min(past_sequence_length+sequence_length,sliding_window-1)",
                                   "2×head_dim", 512, "unresolved_dimension_0"])
        self.assertEqual(out_dims, ["head_dim", "unresolved_dimension_0"])
        snapshot = m.SerializeToString()
        apply_display_dims(m)
        self.assertEqual(snapshot, m.SerializeToString())

    def test_integer_oneof_static_and_dynamic(self):
        set_static_dims({"H": 4096})
        vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["H", "1", "B"])
        g = helper.make_graph([helper.make_node("Identity", ["x"], ["x"])], "g", [vi], [vi])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
        apply_display_dims(m)
        dims = m.graph.input[0].type.tensor_type.shape.dim
        self.assertEqual(dims[0].dim_value, 4096)
        self.assertFalse(dims[0].HasField("dim_param"))
        self.assertEqual(dims[1].dim_value, 1)
        self.assertEqual(dims[2].dim_param, "batch_size")

    def test_numbers_and_unknown_not_guessed(self):
        self.assertEqual(display_dim("512"), 512)
        self.assertEqual(display_dim("head_dim"), "head_dim")
        self.assertEqual(display_dim("unresolved_dimension_7"), "unresolved_dimension_7")


if __name__ == "__main__":
    unittest.main()
