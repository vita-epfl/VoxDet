# [ArXiv 25] VoxDet: Rethinking 3D Semantic Occupancy Prediction as Dense Object Detection

### [[Project Page]](https://vita-epfl.github.io/VoxDet/)  [[Paper (Comming)]]()

📌 This is an official PyTorch implementation of the work:

> [**VoxDet: Rethinking 3D Semantic Occupancy Prediction as Dense Object Detection**]() `<br>`
> [Wuyang Li `<sup>`1 `</sup>`](https://wymancv.github.io/wuyang.github.io/), [Zhu Yu `<sup>`2 `</sup>`](), [Alexandre Alahi `<sup>`1 `</sup>`](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en) `<br><sup>`1 `</sup>` École Polytechnique Fédérale de Lausanne (EPFL); `<sup>`2 `</sup>` Zhejiang University

<div align="center">
    <img width="100%" alt="VoxDet overview" src="assets/introduction.png"/>
</div>


Code is coming soon! We’re currently cleaning up the code and unifying the camera- and LiDAR-based implementations into a single project, which serves as a powerful, clean, and extensible baseline model for the community.

If you can’t wait for the official release, feel free to contact me for the individual implementations.

Contact: [wuyang.li@epfl.ch](mailto:wuyang.li@epfl.ch)

## ✨ Highlight

*VoxDet* address semantic occupancy prediction with an instance-centric formulation inspried by dense object detection, which uses a *Voxel-to-Instance (VoxNT)* trick freely transferring voxel-level class labels to instance-level offset labels.

- **Versatile**: Adaptable to various voxel-based scenarios, such as <span style="color:red">camera and LiDAR</span> settings.
- **Powerful**: Achieves <span style="color:red">joint state-of-the-art</span> on both camera-based and LiDAR-based SSC benchmarks.
- **Efficient**: <span style="color:red">Fast</span> (~1.3× speed-up) and <span style="color:red">Fast</span>lightweight</span> (reducing ~57.9% parameters).
- **Leaderboard Topper**: <span style="color:red">Achieves <span style="color:red">63.0 IoU</span> (single-frame model), securing <span style="color:red">1st</span> place on the SemanticKITTI leaderboard.</span>

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
- [ ] Release all unified codebase, including both camera-based and LiDAR-based implementation
- [ ] Release all models

## 📚Citeation

<div align="center">
    <img width="100%" alt="VoxDet overview" src="assets/overall.png"/>
</div>

If you think our work is helpful for your project, I would greatly appreciate it if you could consdier citing our work

```bibtex
@article{li2025voxdet,
  title={VoxDet: Rethinking 3D Semantic Occupancy Prediction as Dense Object Detection},
  author={Li, Wuyang and Yu, Zhu and Alahi, Alexandre},
  journal={arXiv preprint},
  year={2025}
}
```
