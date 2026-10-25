import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from openpi import transforms
from openpi.models import model as _model


def make_openarm_example() -> dict:
    """Creates a random input example for the OpenArm policy."""
    return {
        "images": {
            "head": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "state": np.zeros(16, dtype=np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Inputs for the OpenArm policy.

    Expected keys (joint-space route):
    - images: {"head": [H,W,3], "left_wrist": [H,W,3], "right_wrist": [H,W,3]}
    - state: [16] = 7 left joints + left gripper + 7 right joints + right gripper
    - actions: [action_horizon, 16]
    - prompt: str
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        base_image = _parse_image(images["head"])
        left_image = _parse_image(images["left_wrist"])
        right_image = _parse_image(images["right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                model_images = (base_image, left_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                model_images = (base_image, left_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": data["state"],
            "image": dict(zip(names, model_images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Outputs for the OpenArm policy (16 joint-space action dims)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :16]}


# ---------------------------------------------------------------------------
# End-effector (Cartesian) route.
#
# State layout (16):   [L p3, q4, grip1,  R p3, q4, grip1]  (world, quat xyzw)
# Absolute action (16): same layout, built from action.eef + gripper commands.
# Delta action (14):   [L dp3, drv3, grip1,  R dp3, drv3, grip1]
#   dp  = p_cmd - p_state
#   drv = rotvec(R(q_cmd) @ R(q_state)^T)   (world-frame relative rotation)
# ---------------------------------------------------------------------------


def make_openarm_ee_example() -> dict:
    """Creates a random input example for the OpenArm EE policy (inference format)."""
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    eef = np.concatenate([
        np.array([0.0, 0.0, 0.5], dtype=np.float32), identity,
        np.array([0.0, 0.0, 0.5], dtype=np.float32), identity,
    ])
    return {
        "images": {
            "head": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "state_eef": eef,
        "state_joint": np.zeros(16, dtype=np.float32),
        "prompt": "do something",
    }


def _image_slots(images: dict, model_type: _model.ModelType):
    base_image = _parse_image(images["head"])
    left_image = _parse_image(images["left_wrist"])
    right_image = _parse_image(images["right_wrist"])

    match model_type:
        case _model.ModelType.PI0 | _model.ModelType.PI05:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            model_images = (base_image, left_image, right_image)
            image_masks = (np.True_, np.True_, np.True_)
        case _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            model_images = (base_image, left_image, right_image)
            image_masks = (np.True_, np.True_, np.True_)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
    return names, model_images, image_masks


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _rot_delta_world(q_cmd: np.ndarray, q_state: np.ndarray) -> np.ndarray:
    """World-frame relative rotation: rotvec(R_cmd @ R_state^T)."""
    q_cmd = _normalize_quat(q_cmd)
    q_state = _normalize_quat(q_state)
    r_cmd = _Rotation.from_quat(q_cmd.reshape(-1, 4))
    q_state_b = np.broadcast_to(q_state, q_cmd.shape).reshape(-1, 4)
    r_state = _Rotation.from_quat(q_state_b)
    return (r_cmd * r_state.inv()).as_rotvec().reshape(q_cmd.shape[:-1] + (3,))


def _rot_apply_world(rotvec: np.ndarray, q_state: np.ndarray) -> np.ndarray:
    """Reconstruct absolute quaternion: R_abs = R_delta @ R_state."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    q_state = _normalize_quat(q_state)
    r_delta = _Rotation.from_rotvec(rotvec.reshape(-1, 3))
    q_state_b = np.broadcast_to(q_state, rotvec.shape[:-1] + (4,)).reshape(-1, 4)
    r_state = _Rotation.from_quat(q_state_b)
    return (r_delta * r_state).as_quat().reshape(rotvec.shape[:-1] + (4,))


# Fixed body-frame transform between the observation eef stream (FK of qpos) and the
# command eef stream (FK of the controller commands). Fitted over all 118 episodes:
#     p_cmd = p_obs + R_obs @ t,   R_cmd = R_obs @ R_t
# Raw gap is ~0.22 m / 0.65 rad; after this correction the dynamic residual is
# ~0.03 m / 0.07 rad, i.e. the actual compliant-arm tracking error.
# Re-fit if the tool/TCP changes.
_EEF_OBS_TO_CMD = (
    (np.array([0.2156, -0.0031, -0.0344]), np.array([0.0088, 0.6453, 0.0306])),
    (np.array([0.2191, -0.0037, -0.0333]), np.array([0.0131, 0.6257, -0.0263])),
)


def _eef_obs_to_cmd(eef: np.ndarray) -> np.ndarray:
    """Map observation-frame EE poses into the command frame (per arm)."""
    out = np.asarray(eef, dtype=np.float64).copy()
    for arm, (t_tool, rotvec) in enumerate(_EEF_OBS_TO_CMD):
        o = arm * 7
        q_obs = _normalize_quat(out[o + 3:o + 7])
        r_obs = _Rotation.from_quat(q_obs)
        out[o:o + 3] = out[o:o + 3] + r_obs.apply(t_tool)
        q_new = (r_obs * _Rotation.from_rotvec(rotvec)).as_quat()
        # Keep the same quaternion hemisphere as the observation stream, which is
        # continuous (checked: no sign flips across all episodes).
        if np.dot(q_new, q_obs) < 0:
            q_new = -q_new
        out[o + 3:o + 7] = q_new
    return out.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class OpenArmEEInputs(transforms.DataTransformFn):
    """Inputs for the OpenArm end-effector policy.

    Expected keys after repacking:
    - images: {"head": [H,W,3], "left_wrist": [H,W,3], "right_wrist": [H,W,3]}
    - state_eef: [14] = L(xyz3+quat4), R(xyz3+quat4), **observation frame**, quat xyzw
    - state_joint: [16] qpos, used to read the two gripper positions
    - actions_eef: [action_horizon, 14] absolute commanded EE poses
    - actions_joint: [action_horizon, 16] joint commands, used to read gripper commands
    - prompt: str

    The default is the raw route: state_eef and actions_eef are used as-is, so the
    delta contains the real command-vs-actual tracking bias. Set
    ``use_bias_correction=True`` only to load legacy checkpoints trained with the
    old mean-bias compensation.
    """

    model_type: _model.ModelType
    use_bias_correction: bool = False

    def __call__(self, data: dict) -> dict:
        names, model_images, image_masks = _image_slots(data["images"], self.model_type)

        eef = np.asarray(data["state_eef"], dtype=np.float32).reshape(14)
        if self.use_bias_correction:
            # Legacy path: mean command-vs-actual bias compensation, not a frame fix.
            eef = _eef_obs_to_cmd(eef)
        qpos = np.asarray(data["state_joint"], dtype=np.float32).reshape(16)
        grip = qpos[[7, 15]]
        # [L p3,q4,grip, R p3,q4,grip]
        state = np.concatenate([eef[:7], grip[:1], eef[7:14], grip[1:2]]).astype(np.float32)

        inputs = {
            "state": state,
            "image": dict(zip(names, model_images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions_eef" in data and "actions_joint" in data:
            actions_eef = np.asarray(data["actions_eef"], dtype=np.float32)
            actions_joint = np.asarray(data["actions_joint"], dtype=np.float32)
            grip_cmd = actions_joint[..., [7, 15]]
            inputs["actions"] = np.concatenate(
                [actions_eef[..., :7], grip_cmd[..., :1], actions_eef[..., 7:14], grip_cmd[..., 1:2]],
                axis=-1,
            ).astype(np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class EEFDeltaActions(transforms.DataTransformFn):
    """Absolute dual-arm EE commands -> world-frame deltas anchored at the current state.

    state (16):   [L p3,q4,grip1, R p3,q4,grip1]
    actions (16): absolute commands, same layout, shape (..., 16)
    output (14):  [L dp3,drv3,grip1, R dp3,drv3,grip1]
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        out = np.empty(actions.shape[:-1] + (14,), dtype=np.float32)

        # (state start, delta output start)
        for s, o in ((0, 0), (8, 7)):
            p_state, q_state = state[s:s + 3], state[s + 3:s + 7]
            p_cmd, q_cmd = actions[..., s:s + 3], actions[..., s + 3:s + 7]

            out[..., o:o + 3] = p_cmd - p_state
            out[..., o + 3:o + 6] = _rot_delta_world(q_cmd, q_state)
            out[..., o + 6] = actions[..., s + 7]  # gripper stays absolute

        data["actions"] = out
        return data


@dataclasses.dataclass(frozen=True)
class EEFAbsoluteActions(transforms.DataTransformFn):
    """Delta EE actions -> absolute EE commands (used by the inference output pipeline).

    state (16):   [L p3,q4,grip1, R p3,q4,grip1]
    actions (14): [L dp3,drv3,grip1, R dp3,drv3,grip1]
    output (16):  [L p3,q4,grip1, R p3,q4,grip1]
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        out = np.empty(actions.shape[:-1] + (16,), dtype=np.float32)

        # (delta start, absolute output start)
        for s, o in ((0, 0), (7, 8)):
            p_state, q_state = state[o:o + 3], state[o + 3:o + 7]
            dp, drv = actions[..., s:s + 3], actions[..., s + 3:s + 6]

            out[..., o:o + 3] = p_state + dp
            out[..., o + 3:o + 7] = _rot_apply_world(drv, q_state)
            out[..., o + 7] = actions[..., s + 6]

        data["actions"] = out
        return data


@dataclasses.dataclass(frozen=True)
class OpenArmEEOutputs(transforms.DataTransformFn):
    """Outputs for the OpenArm EE policy (16 absolute action dims at inference)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :16]}
