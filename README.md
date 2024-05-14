# LLMCarbon
A prelimiary code repo for LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models. More details can be viewed at https://github.com/UnchartedRLab/LLMCarbon. LLMCarbon provides precise predictions of both operational and embodied carbon footprints of large language models (LLMs), enabling effective exploration of the design space by considering the trade-off between test loss and carbon footprint. These carbon footprint exploration can be considered before training an LLM to ensure responsible and sustainable development.

## Run Validations
To generate the data in the table 4 and table 5 in the paper
```
python3 llmcarbon_tutorial.py
```

## Estimation of CO2 equivalent emissions of tranformer based large language models
Estimated regression coefficients used for polynomial fit  $\mathbf{y = ax^2 + bx + c} $
- Tensor model throughput: $$a= -8.82079068\times 10^{-20},  b= 1.68591116\times 10^{-09},  c= 1.33954735\times 10^{+02}$$
- Pipeline model throughput: $$a= -5.60233749\times 10^{-23},  b= 8.45435587\times 10^{-11},  c= 1.34546129\times 10^{+02}$$
- Total number of GPUs: $$a= -2.12910565\times 10^{-21},  b= 4.39684339\times 10^{-09},  c=7.99173057\times 10^{+02}$$
- Batch Size: $$a = -4.29439186\times 10^{-01},  b= 5.21376002\times 10^{+01},  c= 1.43737095\times 10^{+03}$$

![alt text](https://github.com/SotaroKaneda/MLCarbon/blob/main/img/ml_para_set_1.jpg)

## Bibtex

```
@inproceedings{
faiz2024llmcarbon,
title={{LLMC}arbon: Modeling the End-to-End Carbon Footprint of Large Language Models},
author={Ahmad Faiz and Sotaro Kaneda and Ruhan Wang and Rita Chukwunyere Osi and Prateek Sharma and Fan Chen and Lei Jiang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=aIok3ZD9to}
}
```

