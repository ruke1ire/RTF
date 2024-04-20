# Long Range Arena and Synthetics

This directory was directly adapted from the [S4 repository](https://github.com/state-spaces/s4). For detailed explanation on the setup of the python3 environment, datasets, and custom CUDA kernels (for S4), please refer to the original repository.

## Running Experiments

The default configuration for each of the model is located in `configs/model/layer/<model>.yaml`, each LRA and synthetic experiment modifies the default model configuration for the specific task. For LRA, the task-dependent configurations are located in `configs/experiment/lra/<model>-<task>.yaml`.

Run experiments with the following command:
```
python3 -m train experiment=<config path/name>
```

As an example, `python3 -m train experiment=lra/rtf-cifar`, will run the `IMAGE` task in LRA.