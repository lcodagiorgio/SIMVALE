# SIMVALE

### Abstract
The rise of Large Language Model (LLM)-powered simulators has enabled highly realistic modeling of complex social phenomena, significantly reducing the costs and efforts associated with real-world data collection. However, their reliability remains a persistent challenge, and existing validation approaches offer only partial generalization, typically evaluating simulator realism along isolated dimensions and at fixed levels of granularity. To this end, this thesis introduces SIMVALE (SIMulator VAlidation with Latent Embeddings) — a generalizable multi-dimensional validation framework enabling quantitative assessment of LLM-based simulators. SIMVALE leverages latent embedding clustering and yields both global and local feature-based metrics to evaluate how closely a simulator reflects the reality it aims to model, ultimately supporting its improvement. Furthermore, SIMVALE is tested on a real case study to assess the simulator’s ability to replicate general behavioral patterns, toxic profiles, and the effects of moderation interventions.

---

### The framework
SIMVALE is a novel validation framework for LLM-powered simulators that enables their quantitative assessment across multiple behavioral dimensions — which can be tailored to the simulator's modeling goals — and supports evaluations at both coarse, i.e. global, and fine-grained, i.e. local, levels of detail.

The supported validation is two-fold, as it is designed to assess both the simulator's realism, by comparing its generated content to a real-world reference dataset, and the effects of possible alterations on the simulator's output, by comparing its original content to its altered counterpart.

#### But HOW?
The framework is built upon four key phases. First, a transparent feature extraction pipeline is applied to each dataset to extract task-agnostic and domain-specific features, organized into distinct macro-dimensions. Task-agnostic features describe general textual properties, such as linguistic style, sentiment and readability, while domain-specific features are defined according to specific validation objectives, e.g. personality and toxicity.

In the second phase, BERT models are fine-tuned individually on selected domain-specific features using the real data, allowing each model to specialize in detecting linguistic nuances associated with its corresponding target. These fine-tuned models are then employed to derive target-aware, contextualized embeddings for all textual instances in both real and simulated data.

In the third phase, the resulting embeddings are clustered separately for each dataset to uncover homogeneous latent, domain-specific subsets of instances, referred to as _profiles_. These are then aligned across datasets with a similarity-based, one-to-one mapping strategy focusing on the most dominant traits characterizing each profile, allowing for a finer-grained assessment of the simulator by comparing corresponding profile subsets.

Unlike more conventional approaches that would perform clustering directly on — domain-specific — features, with each textual instance represented by a narrow set of scalar values, our method leverages fine-tuned embeddings to uncover richer and more subtle behavioral groupings. Indeed, these vector representations encode not only the general linguistic knowledge acquired during BERT's pre-training, but also the domain-specific patterns learned through the fine-tuning process. As a result, this allows to group instances in a more meaningful, expressive and behaviorally-coherent manner.

Finally, an evaluation pipeline is applied to quantify the distributional divergences across the macro-dimensional features between dataset pairs. To assess realism each real feature is compared with its original simulated counterpart, while to evaluate alteration effects the original simulated features are compared against their altered versions. Features exhibiting significant differences are identified through a statistical test, and further analyzed using a set of comparative metrics. These analyses are conducted both globally, considering the entire datasets, and locally, focusing on the matched profile datasets.

Furthermore, in this work we test the SIMVALE framework on a novel LLM-based OSN simulator, designed to model the emergence of toxic behavior and to evaluate the effects of different moderation interventions on harmful content.

Thus, the validation focuses on assessing the simulator's realism in reproducing real-world OSN patterns, as well as the efficacy of its supported moderation interventions in mitigating toxicity. As discussed earlier, both evaluations are conducted globally, on the entire datasets, and locally, targeting specific toxicity-related content profiles.
