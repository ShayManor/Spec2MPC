from spec2mpc.perception.v2.grasps import add_grasps, grasps_for
from spec2mpc.perception.v2.ground_truth import perceive_gt


def perceive_images(*args, **kwargs):  # lazy: keeps torch/transformers out of the ground-truth path
    from spec2mpc.perception.v2.image_stack import perceive_images as f

    return f(*args, **kwargs)


__all__ = ["add_grasps", "grasps_for", "perceive_gt", "perceive_images"]
