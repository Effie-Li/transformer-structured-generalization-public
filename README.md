# transformer-structured-generalization-public

This repo hosts the model, analysis, and plotting code for:

Li, Y., & McClelland, J.L. (2023). [Representations and Computations in Transformers that Support Generalization on Structured Tasks](https://openreview.net/pdf?id=oFC2LAqS6Z). *Transactions on Machine Learning Research*.

## Abstract

Transformers have shown remarkable success in natural language processing and computer vision, serving as the foundation of large language and multimodal models. These networks can capture nuanced context sensitivity across high-dimensional language tokens or image pixels, but it remains unclear how highly structured behavior and systematic generalization can arise in these systems. Here, we explore the solution process a causal transformer discovers as it learns to solve a set of algorithmic tasks involving copying, sorting, and hierarchical compositions of these operations. We search for the minimal layer and head configuration sufficient to solve these tasks and unpack the roles of the attention heads, as well as how token representations are reweighted across layers to complement these roles. Our results provide new insights into how attention layers in transformers support structured computation within and across tasks: 1) Replacing fixed position labels with labels sampled from a larger set enables strong length generalization and faster learning. The learnable embeddings of these labels develop different representations, capturing sequence order if necessary, depending on task demand. 2) Two-layer transformers can learn reliable solutions to the multi-level problems we explore. The first layer tends to transform the input representation to allow the second layer to share computation across repeated components within a task or across related tasks. 3) We introduce an analysis pipeline that quantifies how the representation space in a given layer prioritizes different aspects of each item. We show that these representations prioritize information needed to guide attention relative to information that only requires downstream readout.

If you use this work, please cite:
```
@article{LiMcClelland2023Representations,
  title={Representations and Computations in Transformers that Support Generalization on Structured Tasks},
  author={Li, Yuxuan and McClelland, James L},
  journal={TMLR},
  year={2023}
}
```
