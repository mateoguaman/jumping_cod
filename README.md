# Jumping CoD (Jumping Continuously over Discontinuities)

This repository contains the code for the paper ["Agile Continuous Jumping over Discontinuous Terrains"](https://yxyang.github.io/jumping_cod).

The main contents of the repo includes:

* The simulation environment to train the terrain-aware jumping controller.
* The code to deploy the trained controller to a real Unitree Go1 robot.
* Additional utilities to inspect robot logs and record data for real-to-sim study.

Please see our [read_the_docs](https://jumping-cod.readthedocs.io/en/latest/) site for detailed documentation.

## CoRL Demo Instructions

To run the evaluation script with jumping boxes, run the following:

```bash
python -m src.agents.heightmap_prediction.eval --logdir=data/box_policy_20240730/box_distill/model_29.pt --save_traj=False --num_envs=1
```

Replace `box_policy_20240730` with one of `[box_policy_20240725, box_policy_20240730, box_policy_20240731, box_policy_20240802]`