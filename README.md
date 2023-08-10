# LLMCarbon
LLMCarbon provides precise predictions of both operational and embodied carbon footprints, enabling effective exploration of the design
space by considering the trade-off between test loss and carbon footprint. These carbon footprint predictions should be considered before training Large Language Models to ensure responsible and sustainable development.

To use MLCarbon TUI
```
python3 mlcarbon_tui.py
```

## Test Validations
To test TFLOP predictions based on # of parameters and # of tokens
```
python3 test/model_test.py
```

To test Throughput and Energy predictions
```
python3 test/training_test.py
```
To test Embodied Carbon predictions

```
python3 test/validate_tables.py
```
## Megatron LM Regression Analysis
![alt text](https://github.com/SotaroKaneda/MLCarbon/blob/main/img/ml_para_set_1.jpg)
## References

Acun, B.; Lee, B.; Kazhamiaka, F.; Maeng, K.; Gupta, U.;
Chakkaravarthy, M.; Brooks, D.; and Wu, C.-J. 2023. Car-
bon explorer: A holistic framework for designing carbon
aware datacenters. In ACM International Conference on Ar-
chitectural Support for Programming Languages and Oper-
ating Systems, Volume 2, 118–132.

Anil, R.; Dai, A. M.; Firat, O.; Johnson, M.; Lepikhin,
D.; Passos, A.; Shakeri, S.; Taropa, E.; Bailey, P.; Chen,
Z.; et al. 2023. Palm 2 technical report. arXiv preprint
arXiv:2305.10403.

Anthony, L. F. W.; Kanding, B.; and Selvan, R. 2020.
Carbontracker: Tracking and predicting the carbon foot-
print of training deep learning models. arXiv preprint
arXiv:2007.03051.

Behnke, M.; and Heafield, K. 2020. Losing heads in the lot-
tery: Pruning transformer attention in neural machine trans-
lation. In the Conference on Empirical Methods in Natural
Language Processing, 2664–2674.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.;
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,
A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan,
T.; Child, R.; Ramesh, A.; Ziegler, D.; Wu, J.; Winter,
C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.;
Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford,
A.; Sutskever, I.; and Amodei, D. 2020. Language Models
are Few-Shot Learners. In Advances in Neural Information
Processing Systems, volume 33, 1877–1901.

Bui, N. D.; Le, H.; Wang, Y.; Li, J.; Gotmare, A. D.;
and Hoi, S. C. 2023. CodeTF: One-stop Transformer
Library for State-of-the-art Code LLM. arXiv preprint
arXiv:2306.00029.

Caballero, E.; Gupta, K.; Rish, I.; and Krueger, D. 2023.
Broken Neural Scaling Laws. In The Eleventh International
Conference on Learning Representations.

Campello de Souza, B.; Serrano de Andrade Neto, A.; and
Roazzi, A. 2023. Are the new ais smart enough to steal your
job? iq scores for chatgpt, microsoft bing, google bard and
quora poe. IQ Scores for ChatGPT, Microsoft Bing, Google
Bard and Quora Poe (April 7, 2023).

Choe, J. 2021. Memory technology 2021: Trends & chal-
lenges. In 2021 International Conference on Simulation of
Semiconductor Processes and Devices (SISPAD), 111–115.
IEEE.

Chowdhery, A.; Narang, S.; Devlin, J.; Bosma, M.; Mishra,
G.; Roberts, A.; Barham, P.; Chung, H. W.; Sutton, C.;
Gehrmann, S.; et al. 2022. Palm: Scaling language modeling
with pathways. arXiv preprint arXiv:2204.02311.

Conneau, A.; Khandelwal, K.; Goyal, N.; Chaudhary, V.;
Wenzek, G.; Guzm ́an, F.; Grave, E.; Ott, M.; Zettlemoyer,
L.; and Stoyanov, V. 2020. Unsupervised Cross-lingual Rep-
resentation Learning at Scale. In Annual Meeting of the As-
sociation for Computational Linguistics, 8440–8451.

Dodge, J.; Prewitt, T.; Tachet des Combes, R.; Odmark, E.;
Schwartz, R.; Strubell, E.; Luccioni, A. S.; Smith, N. A.; De-
Cario, N.; and Buchanan, W. 2022. Measuring the Carbon
Intensity of AI in Cloud Instances. In ACM Conference on
Fairness, Accountability, and Transparency, 1877—-1894.
New York, NY, USA: Association for Computing Machin-
ery. ISBN 9781450393522.

Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch trans-
formers: Scaling to trillion parameter models with simple
and efficient sparsity. The Journal of Machine Learning Re-
search, 23(1): 5232–5270.

Garcia Bardon, M.; Wuytens, P.; Ragnarsson, L.-A.;
Mirabelli, G.; Jang, D.; Willems, G.; Mallik, A.; Spes-
sot, A.; Ryckaert, J.; and Parvais, B. 2020. DTCO
including Sustainability: Power-Performance-Area-Cost-
Environmental score (PPACE) Analysis for Logic Tech-
nologies. In IEEE International Electron Devices Meeting,
41.4.1–41.4.4.

Gupta, U.; Kim, Y. G.; Lee, S.; Tse, J.; Lee, H.-H. S.; Wei,
G.-Y.; Brooks, D.; and Wu, C.-J. 2022. Chasing Carbon:
The Elusive Environmental Footprint of Computing. IEEE
Micro, 42(4): 37—-47.

Henderson, P.; Hu, J.; Romoff, J.; Brunskill, E.; Jurafsky,
D.; and Pineau, J. 2020. Towards the Systematic Reporting
of the Energy and Carbon Footprints of Machine Learning.
Journal of Machine Learning Research, 21(1).

Hoffmann, J.; Borgeaud, S.; Mensch, A.; Buchatskaya, E.;
Cai, T.; Rutherford, E.; Casas, D. d. L.; Hendricks, L. A.;
Welbl, J.; Clark, A.; et al. 2022. Training compute-optimal
large language models. arXiv preprint arXiv:2203.15556.

Jouppi, N. P.; Young, C.; Patil, N.; Patterson, D.; Agrawal,
G.; Bajwa, R.; Bates, S.; Bhatia, S.; Boden, N.; Borchers, A.;
et al. 2017. In-datacenter performance analysis of a tensor
processing unit. In IEEE/ACM International symposium on
computer architecture, 1–12.

Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T. B.;
Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; and
Amodei, D. 2020. Scaling laws for neural language mod-
els. arXiv preprint arXiv:2001.08361.

Lacoste, A.; Luccioni, A.; Schmidt, V.; and Dandres, T.
2019. Quantifying the carbon emissions of machine learn-
ing. arXiv preprint arXiv:1910.09700.

Lieber, O.; Sharir, O.; Lenz, B.; and Shoham, Y. 2021.
Jurassic-1: Technical details and evaluation. White Paper.
AI21 Labs, 1.

Liu, Y.; Wei, X.; Xiao, J.; Liu, Z.; Xu, Y.; and Tian, Y.
2020. Energy consumption and emission mitigation predic-
tion based on data center traffic and PUE for global data
centers. Global Energy Interconnection, 3(3): 272–282.

Narayanan, D.; Shoeybi, M.; Casper, J.; LeGresley, P.; Pat-
wary, M.; Korthikanti, V.; Vainbrand, D.; Kashinkunti, P.;
Bernauer, J.; Catanzaro, B.; Phanishayee, A.; and Zaharia,
M. 2021. Efficient Large-Scale Language Model Training
on GPU Clusters Using Megatron-LM. In ACM Interna-
tional Conference for High Performance Computing, Net-
working, Storage and Analysis.

Patterson, D.; Gonzalez, J.; H ̈olzle, U.; Le, Q.; Liang, C.;
Munguia, L.-M.; Rothchild, D.; So, D. R.; Texier, M.; and
Dean, J. 2022a. The carbon footprint of machine learning
training will plateau, then shrink. Computer, 55(7): 18–28.
Patterson, D.; Gonzalez, J.; H ̈olzle, U.; Le, Q.; Liang, C.;
Munguia, L.-M.; Rothchild, D.; So, D. R.; Texier, M.; and
Dean, J. 2022b. The Carbon Footprint of Machine Learning
Training Will Plateau, Then Shrink. Computer, 55(7): 18–
28.

Patterson, D.; Gonzalez, J.; Le, Q.; Liang, C.; Munguia, L.-
M.; Rothchild, D.; So, D.; Texier, M.; and Dean, J. 2021.
Carbon emissions and large neural network training. arXiv
preprint arXiv:2104.10350.

Rae, J. W.; Borgeaud, S.; Cai, T.; Millican, K.; Hoff-
mann, J.; Song, F.; Aslanides, J.; Henderson, S.; Ring, R.;
Young, S.; et al. 2021. Scaling language models: Methods,
analysis & insights from training gopher. arXiv preprint
arXiv:2112.11446.

Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.;
Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Explor-
ing the Limits of Transfer Learning with a Unified Text-to-
Text Transformer. Journal of Machine Learning Research,
21(140): 1–67.

Sanderson, K. 2023. GPT-4 is here: what scientists think.
Nature, 615(7954): 773.

Scao, T. L.; Fan, A.; Akiki, C.; Pavlick, E.; Ili ́c, S.; Hesslow,
D.; Castagn ́e, R.; Luccioni, A. S.; Yvon, F.; Gall ́e, M.; et al.
2022. Bloom: A 176b-parameter open-access multilingual
language model. arXiv preprint arXiv:2211.05100.

Schwartz, R.; Dodge, J.; Smith, N. A.; and Etzioni, O. 2020.
Green AI. Communications of the ACM, 63(12): 54—-63.

Singh, T.; Rangarajan, S.; John, D.; Schreiber, R.; Oliver,
S.; Seahra, R.; and Schaefer, A. 2020. zen 2: The amd
7nm energy-efficient high-performance x86-64 micropro-
cessor core. In 2020 IEEE International Solid-State Circuits
Conference-(ISSCC), 42–44. IEEE.

Smith, S.; Patwary, M.; Norick, B.; LeGresley, P.; Rajbhan-
dari, S.; Casper, J.; Liu, Z.; Prabhumoye, S.; Zerveas, G.;
Korthikanti, V.; et al. 2022. Using deepspeed and megatron
to train megatron-turing nlg 530b, a large-scale generative
language model. arXiv preprint arXiv:2201.11990.

Strubell, E.; Ganesh, A.; and McCallum, A. 2019a. Energy
and Policy Considerations for Deep Learning in NLP. In
Annual Meeting of the Association for Computational Lin-
guistics, 3645–3650.

Strubell, E.; Ganesh, A.; and McCallum, A. 2019b. Energy
and Policy Considerations for Deep Learning in NLP. In
Annual Meeting of the Association for Computational Lin-
guistics, 3645–3650. Florence, Italy: Association for Com-
putational Linguistics.

Thompson, N. C.; Greenewald, K.; Lee, K.; and Manso,
G. F. 2021. Deep Learning’s Diminishing Returns: The Cost
of Improvement is Becoming Unsustainable. IEEE Spec-
trum, 58(10): 50–55.

Thoppilan, R.; De Freitas, D.; Hall, J.; Shazeer, N.; Kul-
shreshtha, A.; Cheng, H.-T.; Jin, A.; Bos, T.; Baker, L.; Du,
Y.; et al. 2022. Lamda: Language models for dialog appli-
cations. arXiv preprint arXiv:2201.08239.

TSMC. 2019. TSMC Corporate Social Responsibility Re-
port. https://esg.tsmc.com/download/file/2019-csr-report/
english/pdf/e-all.pdf.

Wei, J.; Bosma, M.; Zhao, V.; Guu, K.; Yu, A. W.; Lester, B.;
Du, N.; Dai, A. M.; and Le, Q. V. 2022. Finetuned Language
Models are Zero-Shot Learners. In International Conference
on Learning Representations.

Wiki. 2023a. Ampere (microarchitecture).
http://en.wikipedia.org/w/index.php?title=Ampere\
%20(microarchitecture)&oldid=1160464393.

Wiki. 2023b. Tensor Processing Unit. http://en.wikipedia.
org/w/index.php?title=Tensor\%20Processing\%20Unit&
oldid=1158650479.

Wu, C.-J.; Raghavendra, R.; Gupta, U.; Acun, B.; Ardalani,
N.; Maeng, K.; Chang, G.; Aga, F.; Huang, J.; Bai, C.; et al.
2022. Sustainable ai: Environmental implications, chal-
lenges and opportunities. Proceedings of Machine Learning
and Systems, 4: 795–813.

Xiao, G.; Lin, J.; Seznec, M.; Wu, H.; Demouth, J.; and
Han, S. 2023. SmoothQuant: Accurate and Efficient Post-
Training Quantization for Large Language Models. In In-
ternational Conference on Machine Learning.

Xing, E. P.; Ho, Q.; Dai, W.; Kim, J. K.; Wei, J.; Lee, S.;
Zheng, X.; Xie, P.; Kumar, A.; and Yu, Y. 2015. Petuum:
A New Platform for Distributed Machine Learning on Big
Data. IEEE Transactions on Big Data, 1(2): 49–67.

Yandex. 2022. YaLM 100B. https://github.com/yandex/
YaLM-100B.

Zeng, A.; Liu, X.; Du, Z.; Wang, Z.; Lai, H.; Ding, M.; Yang,
Z.; Xu, Y.; Zheng, W.; Xia, X.; Tam, W. L.; Ma, Z.; Xue, Y.;
Zhai, J.; Chen, W.; Liu, Z.; Zhang, P.; Dong, Y.; and Tang,
J. 2023. GLM-130B: An Open Bilingual Pre-trained Model.
In The Eleventh International Conference on Learning Rep-
resentations.
