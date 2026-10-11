from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "data/metrica/data"
    pff_dir: str = "data/pff/cache"
    skillcorner_dir: str = "data/skillcorner/cache"
    dfl_dir: str = "data/dfl/cache"
    # "metrica:<game>", "pff:<id>", "pff:train" (all but held-out), "pff:val" (every 11th id, held out)
    train_games: tuple = ("pff:train", "metrica:Sample_Game_1")
    val_games: tuple = ("pff:val",)
    # Reported every epoch with the simulated camera but never used to pick the checkpoint (e.g. other providers).
    extra_val: tuple = ()
    max_val_windows: int = 3000

    # Broadcast-camera simulation on the observed history (targets stay complete -> learn to infer off-screen).
    cam_prob: float = 0.7
    cam_width_min: float = 30.0
    cam_width_max: float = 60.0
    det_drop: float = 0.05
    ball_drop: float = 0.1

    # Time resolution: one model step = step_s seconds. Raise it if the ETA is too long.
    step_s: float = 4.0
    hist_steps: int = 16
    fut_steps: int = 8
    min_context_s: float = 600.0

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    n_modes: int = 6
    dropout: float = 0.1

    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 30
    samples_per_epoch: int = 12000

    w_pos: float = 0.1
    w_vel: float = 0.1
    w_mode: float = 0.2
    w_poss: float = 1.0
    w_state: float = 0.5
    w_event: float = 1.0
    # "hard": one-hot possession label (within 1.5 m of a ball slower than 10 m/s, else nobody, ~70% of labels);
    # "soft": each player's degree of control, falling with distance to the ball and with ball speed.
    poss_target: str = "hard"
    ctrl_sigma_m: float = 1.5

    device: str = "auto"
    seed: int = 0
    ckpt: str = "checkpoints/h2_pff_cam.pt"
    # Checkpoint selection on the camera-occluded val windows. ADE_top1 scores the future the model itself picks (what
    # the UI shows); minADE scores the best of K futures chosen with hindsight (the h2_pff_cam.pt run used minADE).
    select_metric: str = "ADE_top1"
    log_every: int = 25
