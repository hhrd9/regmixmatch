## Acknowledgements

This project incorporates code from [TorchSSL](https://github.com/TorchSSL/TorchSSL), which is licensed under the MIT License.

The following is the full text of the MIT License as applied to TorchSSL:
MIT License

Copyright (c) 2021 TorchSSL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Usage
This is an PyTorch implementation of RegMixMatch. Note that all our baseline is based on the implementation of TorchSSL framework. We would like to thank the authors of this repository.


Before running or modifing the code, you need to:
1. Clone this repo to your machine.
2. Make sure Anaconda or Miniconda is installed.
3. Run `conda env create -f environment.yml` for environment initialization.

### Run the experiments
As introduced in paper, we implement RegMixMatch based on FreeMatch [1].
If you want to run RegMixMatch algorithm:

1. Modify the config file in `config/freematch_entropy/freematch_entropy_xx_xx_xx.yaml` as you need
2. Run `python freematch_entropy.py --c config/freematch_entropy/freematch_entropy_xx_xx_xx.yaml`

## References


[1] Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernt Schiele, Xing Xie. FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning. ICLR, 2023.
