# RTF
This repository contains the official implementation of RTF state-space model.

## Repository Structure

- [`rtf.py`](rtf.py) contains the standalone RTF implementation.
- Experiemnts are located in `experiments`.
	- [`experiments/safari`](experiments/safari): Wikitext103 language modeling experiment. 
	- [`experiments/state-spaces`](experiments/state-spaces): Long Range Arena (LRA), synthetic memory tasks (*Copying* and *Delay*), Speech Commands. 
- For each experimental framework, we implement a wrap of the standalone RTF (`rtf.py`).
	- Wrapper for `state-spaces`: [`experiments/state-spaces/src/models/sequence/kernels/rtf_wrapper.py`](experiments/state-spaces/src/models/sequence/kernels/rtf_wrapper.py)
	- Wrapper for `safari`: [`experiments/safari/src/models/sequence/ssm/rtf_wrapper.py`](experiments/safari/src/models/sequence/ssm/rtf_wrapper.py)

## Setup and Usage Guides

Experiment-specific setup and usage guides:
- `state-spaces`: [`experiments/state-spaces/README.md`](experiments/state-spaces/README.md)
- `safari`: [`experiments/safari/README.md`](experiments/safari/README.md)

Setup for standalone rtf.py:
```
pip3 install -r requirements.txt
```

### Example Usage

```python
from rtf import RTF
import torch

seq_len = 1024
d_model = 32
batch_size = 1
input = torch.rand(batch_size, seq_len, d_model)

model = RTF(d_model=d_model, state_size=128, trunc_len=seq_len)

output = model(input)
print(output.shape)
>>> torch.Size([1, 1024, 32])
```
