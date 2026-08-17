# ecg3kg

Physiologically-inspired 3D augmentation of 12-lead ECGs, after **3KG** (Gopal et al., ML4H 2021): the twelve leads are projections of a single 3D cardiac dipole (the vectorcardiogram, VCG); we reconstruct that dipole, rotate and scale it, and project it back to the leads.

> Work in progress — API and README will be completed before v0.1.0.

## Reference

```bibtex
@inproceedings{gopal2021_3kg,
  title     = {3KG: Contrastive Learning of 12-Lead Electrocardiograms using Physiologically-Inspired Augmentations},
  author    = {Gopal, Bryan and Han, Ryan and Raghupathi, Gautham and Ng, Andrew and Tison, Geoff and Rajpurkar, Pranav},
  booktitle = {Proceedings of Machine Learning for Health (ML4H)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {158},
  pages     = {156--167},
  year      = {2021},
  url       = {https://proceedings.mlr.press/v158/gopal21a.html}
}
```

## License

MIT — see [LICENSE](LICENSE).
