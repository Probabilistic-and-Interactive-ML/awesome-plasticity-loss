# Awesome Plasticity Loss

A collection of papers and codebases on plasticity loss accompanying [our survey on the topic](https://arxiv.org/abs/2411.04832), with a focus on deep Reinforcement Learning.
Papers are categorized by their approach to remedy plasticity loss.
Inspired by similar repos on [self-supervised-learning](https://github.com/jason718/awesome-self-supervised-learning) and
[in-context Reinforcement Learning](https://github.com/dunnolab/awesome-in-context-rl).

> [!CAUTION]
>
> ### Repo vs survey state
>
> We aim to keep this repository up-to-date with the latest research on plasticity loss. This is more difficult for the accompanying survey, which is a snapshot of the field at the time of writing. If you are looking for the most recent research, please refer to the papers in this repository.

## Contributing

Feel free to contribute either with a PR or by opening an issue.

Format for papers:

```markdown
- Paper Name.
  [[pdf]](link)
  [[code]](link)
  - Author 1, Author 2, and Author 3. *Conference Year*
```

## Table of Contents

- [Weight Resets](#weight-resets)
  - [Non-targeted](#non-targeted)
  - [Targeted](#targeted)
- [Parameter Regularization](#parameter-regularization)
- [Feature Rank Regularization](#feature-rank-regularization)
- [Activation Functions](#activation-functions)
- [Categorical Losses](#categorical-losses)
- [Distillation](#distillation)
- [Architectures](#architectures)
- [Other Approaches](#other-approaches-and-papers)
- [Combined Methods](#combined-methods)

## Weight Resets

General resetting algorithms.

### Non-targeted

Reset parameters of the network irrespective of their utility to the agent.

- The Primacy Bias in Deep Reinforcement Learning.
  [[pdf]](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf)
  [[code]](https://github.com/evgenii-nikishin/rl_with_resets)
  - Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, Aaron Courville. *ICML 2022*
- Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier.
  [[pdf]](https://openreview.net/pdf?id=OpC-9aBBVJe)
  [[code]](https://github.com/proceduralia/high_replay_ratio_continuous_control)
  - Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G Bellemare, Aaron Courville. *ICLR 2023*
- DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization.
  [[pdf]](https://arxiv.org/pdf/2310.19668)
  [[code]](https://drm-rl.github.io/)
  - Guowei Xu, Ruijie Zheng, Yongyuan Liang, Xiyao Wang, Zhecheng Yuan, Tianying Ji, Yu Luo, Xiaoyu Liu, Jiaxin Yuan, Pu Hua, Shuzhen Li, Yanjie Ze, Hal Daumé III, Furong Huang, Huazhe Xu. *ICLR 2024*

### Targeted

Reset specific parameters/neurons of the network, usually based on some measure of utility.

- Loss of plasticity in deep continual learning.
  [[pdf]](https://www.nature.com/articles/s41586-024-07711-7)
  [[code]](https://github.com/shibhansh/loss-of-plasticity)
  - Shibhansh Dohare, J Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A Rupam Mahmood, Richard S Sutton. *Nature 2024*
- The Dormant Neuron Phenomenon in Deep Reinforcement Learning.
  [[pdf]](https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf)
  [[code (jax)]](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)
  [[code (pytorch)]](https://github.com/timoklein/redoo)
  - Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci. *ICML 2023*
- Addressing loss of plasticity and catastrophic forgetting in continual learning.
  [[pdf]](https://arxiv.org/pdf/2404.00781)
  [[code]](https://github.com/mohmdelsayed/upgd)
  - Mohamed Elsayed, A Rupam Mahmood. *ICLR 2024*
- Deep Reinforcement Learning with Plasticity Injection.
  [[pdf]](https://proceedings.NeurIPS.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf)
  [[code]](https://github.com/timoklein/plasticity-injection-torch)
  - Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, Andre Barreto. *NeurIPS 2023*

## Parameter Regularization

Regularize the parameters of the network towards values that are less prone to plasticity loss.

- Maintaining Plasticity in Continual Learning via Regenerative Regularization.
  [[pdf]](https://arxiv.org/pdf/2308.11958)
  [[code]](https://github.com/skumar9876/L2_Init)
  - Saurabh Kumar, Henrik Marklund, Benjamin Van Roy.
- Towards Deeper Deep Reinforcement Learning with Spectral Normalization.
  [[pdf]](https://arxiv.org/pdf/2106.01151)
  - Johan Bjorck, Carla P. Gomes, Kilian Q. Weinberger. *NeurIPS 2021*
- Spectral Normalisation for Deep Reinforcement Learning: an Optimisation Perspective.
  [[pdf]](https://arxiv.org/pdf/2105.05246)
  [[code]](https://github.com/floringogianu/snrl)
  - Florin Gogianu, Tudor Berariu, Mihaela Rosca, Claudia Clopath, Lucian Busoniu, Razvan Pascanu. *ICML 2021*
- Weight Clipping for Deep Continual and Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2407.01704)
  [[code]](https://github.com/mohmdelsayed/weight-clipping)
  - Mohamed Elsayed, Qingfeng Lan, Clare Lyle, A. Rupam Mahmood. *RLC 2024*
- Directions of Curvature as an Explanation for Loss of Plasticity.
  [[pdf]](https://arxiv.org/pdf/2312.00246)
  - Alex Lewandowski, Haruto Tanaka, Dale Schuurmans, Marlos C. Machado.
- Learning Continually by Spectral Regularization.
  [[pdf]](http://arxiv.org/pdf/2406.06811)
  - Alex Lewandowski, Saurabh Kumar, Dale Schuurmans, András György, Marlos C. Machado.

## Feature Rank Regularization

Regularize the rank of the feature matrix either directly or indirectly.

- Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning.
  [[pdf]](https://openreview.net/pdf?id=O9bnihsFfXU)
  [[code]](https://github.com/timoklein/implicit_underparameterization)
  - Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine. *ICLR 2021*
- Understanding and Preventing Capacity Loss in Reinforcement Learning.
  [[pdf]](https://openreview.net/pdf?id=ZkC8wKoLbQ7)
  [[code]](https://github.com/timoklein/infer)
  - Clare Lyle, Mark Rowland, Will Dabney. *ICLR 2022*
- DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization.
  [[pdf]](https://openreview.net/pdf?id=POvMvLi91f)
  - Aviral Kumar, Rishabh Agarwal, Tengyu Ma, Aaron Courville, George Tucker, Sergey Levine. *ICLR 2022*
- An empirical study of implicit regularization in deep offline RL.
  [[pdf]](https://openreview.net/pdf?id=HFfJWx60IT)
  - Caglar Gulcehre, Srivatsan Srinivasan, Jakub Sygnowski, Georg Ostrovski, Mehrdad Farajtabar, Matthew Hoffman, Razvan Pascanu, Arnaud Doucet. *TMLR 2022*
- Adaptive Regularization of Representation Rank as an Implicit Constraint of Bellman Equation.
  [[pdf]](https://arxiv.org/pdf/2404.12754v1)
  [[code]](https://github.com/sweetice/BEER-ICLR2024)
  - Qiang He, Tianyi Zhou, Meng Fang, Setareh Maghsudi. *ICLR 2024*

## Activation Functions

Proposed activation functions that mitigate plasticity loss.

- An Evaluation of Parametric Activation Functions for Deep Learning.
  [[pdf]](https://ieeexplore.ieee.org/abstract/document/8913972)
  - Luke B. Godfrey. *IEEE SMC 2019*
- Loss of Plasticity in Continual Deep Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2303.07507)
  - Zaheer Abbas, Rosie Zhao, Joseph Modayil, Adam White, Marlos C. Machado. *CoLLAs 2023*
- Adaptive Rational Activations to Boost Deep Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2102.09407)
  [[code]](https://github.com/k4ntz/activation-functions)
  - Quentin Delfosse, Patrick Schramowski, Martin Mundt, Alejandro Molina, Kristian Kersting. *ICLR 2024*
- Hadamard Representations: Augmenting Hyperbolic Tangents in RL
  [[pdf]](https://arxiv.org/pdf/2406.09079v2)
  - Jacob E. Kooi, Mark Hoogendoorn, Vincent François-lavet
- Plastic Learning with Deep Fourier Features.
  [[pdf]](http://arxiv.org/pdf/2410.20634)
  - Alex Lewandowski, Dale Schuurmans, Marlos C. Machado.

## Categorical Losses

Project the scalar regression targets onto a categorical distribution to apply a cross-entropy loss.

- Improving Regression Performance with Distributional Losses.
  [[pdf]](https://proceedings.mlr.press/v80/imani18a/imani18a.pdf)
  [[code]](https://github.com/marthawhite/Histogram_loss)
  - Ehsan Imani, Martha White. *ICML 2018*
- Stop Regressing: Training Value Functions via Classification for Scalable Deep RL.
  [[pdf]](https://openreview.net/pdf?id=dVpFKfqF3R)
  - Jesse Farebrother, Jordi Orbay, Quan Vuong, Adrien Ali Taiga, Yevgen Chebotar, Ted Xiao, Alex Irpan, Sergey Levine, Pablo Samuel Castro, Aleksandra Faust, Aviral Kumar, Rishabh Agarwal . *ICML 2024*

## Distillation

Periodically distill the knowledge of the network into a fresh network.

- Transient Non-Stationarity and Generalisation in Deep Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2006.05826)
  [[code]](https://github.com/maximilianigl/rl-iter)
  - Maximilian Igl, Gregory Farquhar, Jelena Luketina, Wendelin Boehmer, Shimon Whiteson. *ICLR 2021*
- Slow and Steady Wins the Race Maintaining Plasticity with Hare and Tortoise Networks.
  [[pdf]](https://openreview.net/pdf?id=ZkC8wKoLbQ7)
  [[code]](https://github.com/dojeon-ai/hare-tortoise)
  - Hojoon Lee, Hyunseo Cho, Donghu Kim, Hyunseung Kim, Dukgi Min, Jaegul Choo, and Clare Lyle. *ICML 2024*

## Architectures

Specific architectures or interventions on the architecture (e.g., pruning) that mitigate plasticity loss.

- Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control.
  [[pdf]](https://arxiv.org/pdf/2405.16158v1)
  [[code]](https://github.com/naumix/BiggerRegularizedOptimistic)
  - Michal Nauman, Mateusz Ostaszewski, Krzysztof Jankowski, Piotr Miłoś, Marek Cygan. *NeurIPS 2024*
- SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning.
  [[pdf]](http://arxiv.org/pdf/2410.09754)
  [[code]](https://github.com/SonyResearch/simba)
  - Hojoon Lee, Dongyoon Hwang, Donghu Kim, Hyunseung Kim, Jun Jet Tai, Kaushik Subramanian, Peter R. Wurman, Jaegul Choo, Peter Stone, Takuma Seno.
- In value-based deep reinforcement learning, a pruned network is a good network.
  [[pdf]](http://arxiv.org/pdf/2402.12479)
  [[code]](https://github.com/google/dopamine/tree/master/dopamine/)
  - Johan Obando-Ceron, Aaron Courville, Pablo Samuel Castro. *ICML 2024*
- Neuroplastic Expansion in  Deep Reinforcement Learning.
  [[pdf]](http://arxiv.org/pdf/2410.07994)
  - Jiashun Liu, Johan Obando-Ceron, Aaron Courville, Ling Pan.
- Mixtures of Experts Unlock Parameter Scaling for Deep RL.
  [[pdf]](http://arxiv.org/pdf/2402.08609)
  [[code]](https://github.com/google/dopamine/tree/master/dopamine/labs/moes)
  - Johan Obando-Ceron, Ghada Sokar, Timon Willi, Clare Lyle, Jesse Farebrother, Jakob Foerster, Gintare Karolina Dziugaite, Doina Precup, Pablo Samuel Castro. *ICML 2024*

## Other Approaches and Papers

Methods that do not fit in the previous categories.

- Is High Variance Unavoidable in RL? A Case Study in Continuous Control.
  [[pdf]](https://arxiv.org/pdf/2110.11222)
  - Johan Bjorck, Carla P. Gomes, Kilian Q. Weinberger. *ICLR 2022*
- Resetting the Optimizer in Deep RL: An Empirical Study.
  [[pdf]](https://arxiv.org/pdf/2306.17833)
  - Kavosh Asadi, Rasool Fakoor, Shoham Sabach. *NeurIPS 2023*
- Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages.
  [[pdf]](https://openreview.net/pdf?id=0aR1s9YxoL)
  [[code]](https://github.com/Guozheng-Ma/Adaptive-Replay-Ratio)
  - Guozheng Ma, Lu Li, Sen Zhang, Zixuan Liu, Zhen Wang, Yixin Chen, Li Shen, Xueqian Wang, Dacheng Tao. *ICLR 2024*
- Sharpness-Aware Minimization for Efficiently Improving Generalization.
  [[pdf]](https://arxiv.org/pdf/2010.01412)
  [[code]](https://github.com/google-research/sam)
  - Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. *ICLR 2021*
- Harnessing Discrete Representations For Continual Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2312.01203)
  [[code]](https://github.com/ejmejm/discrete-representations-for-continual-rl)
  - Edan Meyer, Adam White, Marlos C. Machado.
- A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning.
  [[pdf]](http://arxiv.org/pdf/2405.19153)
  [[code]](https://github.com/awjuliani/deep-rl-plasticity)
  - Arthur Juliani, Jordan T. Ash. *NeurIPS 2024*

## Combined Methods

Combinations of the previous methods.

- PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2306.10711)
  [[code]](https://github.com/dojeon-ai/plastic)
  - Methods: SAM, LayerNorm, CReLU, Hard head resets
  - Hojoon Lee, Hanseul Cho, Hyunseung Kim, Daehoon Gwak, Joonkee Kim, Jaegul Choo, Se-Young Yun, Chulhee Yun. *NeurIPS 2023*
- Bigger, Better, Faster: Human-level Atari with human-level efficiency.
  [[pdf]](https://arxiv.org/pdf/2305.19452)
  [[code]](https://github.com/google-research/google-research/tree/master/bigger_better_faster)
  - Methods: Shrink & Perturb CNN resets, Hard head resets, Weight decay
  - Max Schwarzer, Johan Obando-Ceron, Aaron Courville, Marc Bellemare, Rishabh Agarwal, Pablo Samuel Castro. *ICML 2023*
- Overestimation, Overfitting, and Plasticity in Actor-Critic: the Bitter Lesson of Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2403.00514)
  - Methods: LayerNorm, Weight decay / Hard resets, Weight decay / LayerNorm, Hard resets
  - Michal Nauman, Michał Bortkiewicz, Piotr Miłoś, Tomasz Trzciński, Mateusz Ostaszewski, Marek Cygan. *ICML 2024*
- Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control.
  [[pdf]](https://arxiv.org/pdf/2405.16158v1)
  [[code]](https://github.com/naumix/BiggerRegularizedOptimistic)
  - Methods: Hard resets, LayerNorm, Weight decay
  - Michal Nauman, Mateusz Ostaszewski, Krzysztof Jankowski, Piotr Miłoś, Marek Cygan. *NeurIPS 2024*
- Disentangling the Causes of Plasticity Loss in Neural Networks.
  [[pdf]](https://arxiv.org/pdf/2402.18762)
  - Methods: LayerNorm, Weight decay
  - Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Hado van Hasselt, Razvan Pascanu, James Martens, Will Dabney.
- Normalization and effective learning rates in reinforcement learning.
  [[pdf]](https://arxiv.org/pdf/2407.01800)
  - Methods: LayerNorm, Parameter regularization
  - Clare Lyle, Zeyu Zheng, Khimya Khetarpal, James Martens, Hado van Hasselt, Razvan Pascanu, Will Dabney.
- Understanding plasticity in neural networks.
  [[pdf]](https://arxiv.org/pdf/2303.01486)
  - Methods: LayerNorm, Weight decay, Categorical loss
  - Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney. *ICML 2023*

## Citation

If you find this repository useful, please consider citing our survey paper:

```bibtex
@article{klein2024plasticity,
  title={Plasticity Loss in Deep Reinforcement Learning: A Survey},
  author={Klein, Timo and Miklautz, Lukas and Sidak, Kevin and Plant, Claudia and Tschiatschek, Sebastian},
  journal={arXiv e-prints},
  pages={arXiv--2411},
  year={2024}
}
```
