import tensorflow as tf

try:
    tf.compat.v1.disable_eager_execution()
except RuntimeError:
    pass

import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Element-wise real division using TensorFlow RealDiv.
    tf.raw_ops.RealDiv(x=x1, y=x2) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: float, fusion_mode: str = 'float32') -> torch.Tensor:
        """
        Applies element-wise real division via TensorFlow.

        Args:
            x1 (torch.Tensor): Dividend tensor.
            x2 (float): Divisor scalar.
            fusion_mode (str, optional): Data type mode. Default: 'float32'.

        Returns:
            torch.Tensor: Output tensor of division result.
        """
        tf.compat.v1.reset_default_graph()

        x1_np = x1.detach().cpu().numpy()

        x1_ph = tf.compat.v1.placeholder(x1_np.dtype, shape=x1_np.shape)
        x2_const = tf.constant(x2, dtype=x1_np.dtype)
        out = tf.raw_ops.RealDiv(x=x1_ph, y=x2_const)

        feed_dict = {x1_ph: x1_np}
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

        with tf.compat.v1.Session(config=session_config) as sess:
            outputs = sess.run(out, feed_dict=feed_dict)

        return torch.from_numpy(outputs)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_RealDiv.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        fusion_mode_info = inputs[2]

        x1 = torch.randn(x1_info["shape"], dtype=torch.float32)
        x2 = float(x2_info["value"])
        fusion_mode = fusion_mode_info["value"]
        input_groups.append([x1, x2, fusion_mode])
    return input_groups


def get_init_inputs():
    return []
