# [ArXiv 25] VoxDet: Rethinking 3D Semantic Scene Completion as Dense Object Detection


### [[Project Page]](https://vita-epfl.github.io/VoxDet/)  [[Paper (Comming)]]()  

📌 This is an official PyTorch implementation of VoxDet: Rethinking 3D Semantic Scene Completion as Dense Object Detection

> [**VoxDet: Rethinking 3D Semantic Scene Completion as Dense Object Detection**]()<br>
> [Wuyang Li<sup>1</sup>](https://wymancv.github.io/wuyang.github.io/), [Zhu Yu<sup>2</sup>](), [Alexandre Alahi<sup>1</sup>](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)<br><sup>1</sup> École Polytechnique Fédérale de Lausanne (EPFL); <sup>2</sup> Zhejiang University
 
<div align="center">
    <img width="100%" alt="VoxDet overview" src="assets/introduction.png"/>
</div>

Contact: [wuyang.li@epfl.ch](mailto:wuyang.li@epfl.ch)

## ✨ Highlight

*VoxDet* address SSC with an instance-centric formulation inspried by dense object detection, which uses a *Voxel-to-Instance (VoxNT)* trick freely transferring voxel-level class labels to instance-level offset labels. More details can be found in our [[Project Page]](https://vita-epfl.github.io/VoxDet/).

- Versatile: Adaptable to various voxel-based scenarios, such as camera and LiDAR applications.
- Powerful: Achieves state-of-the-art on both camera-based and LiDAR-based SSC benchmarks.
- Efficient: Fast (~1.3× speed-up) and lightweight (reducing ~57.9% parameters).
- Leaderboard Topper: Achieving 63.0 IoU (single-frame model), securing 1st place on the SemanticKITTI leaderboard.


Note that VoxDet is single-frame single-model method without extra data and labels.
<div align="center">
    <img width="100%" alt="VoxDet overview" src="assets/leaderboard.png"/>
</div>

## 🙏 Acknowledgement

Greatly appreciate the tremendous effort for the following projects!
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) 
- [Context and Geometry Aware Voxel Transformer for Semantic Scene Completion](https://arxiv.org/abs/2405.13675)
- [SIGMA: Semantic-complete Graph Matching For Domain Adaptive Object Detection](https://arxiv.org/abs/2203.06398)
- [Revisiting the Sibling Head in Object Detector](https://arxiv.org/abs/2003.07540)
- [VoxFormer: a Cutting-edge Baseline for 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2302.12251)

## 📋 TODO List
- [ ] Release the paper
- [ ] Release all code, including both camera-based and LiDAR-based implementation
- [ ] Release all models


##  📚Citeation

<div align="center">
    <img width="100%" alt="VoxDet overview" src="assets/overall.png"/>
</div>

If you think our work is helpful for your project, I would greatly appreciate it if you could consdier citing our work

```bibtex
@article{li2025voxdet,
  title={VoxDet: Rethinking 3D Semantic Scene Completion as Dense Object Detection},
  author={Li, Wuyang and Yu, Zhu and Alahi, Alexandre},
  journal={arXiv preprint},
  year={2025}
}
```




