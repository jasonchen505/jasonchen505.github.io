---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I'm a master's student in Computer Science at Zhejiang University. I received my bachelor's degree in Computer Engineering from Zhejiang University in 2025. My research interests include large language models, agents, training and inference systems, search and recommendation, distributed learning, and AI safety and privacy. My approach to research begins with observing phenomena, exploring possible explanations, developing theories, and testing their implications to arrive at solutions or methods. I'm always happy to connect with others who share these interests or a similar approach to research.



# 📝 Publications 
(\*: Equal Contribution; ${}^\dagger$: Corresponding Author)

## Strategic Unlearnable Example Portfolios for Accountable Image Privacy
**Yizhou Chen**, Meng Zhang
- ECCVW2026

## Personalized Local Differentially Private Federated Learning with Adaptive Client Sampling
**Yizhou Chen**\*, Wangjie Xu\*, Xincheng Wu\*, Meng Zhang, Bing Luo
-  [[ICASSP2024](https://ieeexplore.ieee.org/abstract/document/10447542)][[Paper](https://github.com/jasonchen505/AdaDPFedAvg/blob/main/icassp2024.pdf)]



# 💻 Internships
- *2025.12 - 2026.05*, Agent Training and Multimodal Inference Intern at Tencent Youtu Lab.
- *2023.06 - 2023.12*, Research Intern at DCD, Zhejiang University.    Advised by Prof. Yin Zhang.

# 🛠️ Open Source {#opensource}

- **NVIDIA NeMo Molt** ([#197](https://github.com/NVIDIA-NeMo/labs-molt/pull/197)): Adapted DAPO and ProRL reward shaping for multi-turn agents.
- **Tencent Hunyuan UniRL** ([#631](https://github.com/Tencent-Hunyuan/UniRL/pull/631)): Fixed vLLM-Omni rollout logprob handling.
- **AgenticArXiv-RL** ([#113](https://github.com/Algorineko/AgenticArXiv-RL/pull/113), [#114](https://github.com/Algorineko/AgenticArXiv-RL/pull/114)): Fixed tool-result scoring and zero-variance reward classification in GRPO.
- **vLLM Speculators** ([#1200](https://github.com/vllm-project/speculators/pull/1200)): Improved vocabulary-mapping utility correctness.
- **Alibaba InferSim** ([#20](https://github.com/alibaba/InferSim/pull/20)): Fixed attention MFU configuration for MLA prefill simulation.
- **MLC-AI pith-train** ([#112](https://github.com/mlc-ai/pith-train/pull/112)): Clarified the limitations of FP8 memory estimates.
