# How to Trust Your Diffusion Model:<br /> A Convex Optimization Approach to Conformal Risk Control

[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.7990217.svg)](https://zenodo.org/record/7990217)

This is the official implementation of the paper [*How To Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control*](https://arxiv.org/abs/2302.03791) @ [ICML 2023](https://icml.cc/virtual/2023/paper_metadata_from_author/24442)

by [Jacopo Teneggi](https://jacopoteneggi.github.io), [Matt Tivnan](https://scholar.google.com/citations?user=G2Cc0okAAAAJ&hl=en), [J Webster Stayman](https://scholar.google.com/citations?user=pn8ZDP4AAAAJ&hl=en), and [Jeremias Sulam](https://sites.google.com/view/jsulam).

---

$K$-RCPS is a high-dimensional extension of the [Risk Controlling Prediction Sets (RCPS)](https://github.com/aangelopoulos/rcps) procedure that provably minimizes the mean interval length by means of a convex relaxation.

It is based on $\ell^{\gamma}$: a convex upper-bound to the $01$ loss $\ell^{01}$

<p align="center">
  <img width="460" src="assets/loss.jpg">
</p>

## Demo

The demo is included in the `demo.ipynb` notebook. It showcases how to use the $K$-RCPS calibration procedure on dummy data.

<p align="center">
  <img src="assets/results.gif">
</p>

which reduces the mean interval length compared to RCPS on the same data by $\approx 9$%.

## Reproducibility

All model checkpoints are available on [Zenodo](https://zenodo.org/record/7990217) alongside the perturbed images used in the paper. `checkpoints.zip` and `denoising.zip` should both be unzipped in the `experiments` folder.

## References
```
@article{teneggi2023trust,
  title={How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control},
  author={Teneggi, Jacopo and Tivnan, Matt and Stayman, J Webster and Sulam, Jeremias},
  journal={arXiv preprint arXiv:2302.03791},
  year={2023}
}
```