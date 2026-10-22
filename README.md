# low-level-introspection

<p align="left">
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-b31b1b.svg?style=flat&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  <a href="https://github.com/PootieT/low-level-introspection/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/PootieT/low-level-introspection" alt="License"/>
  </a>
</p>

This repo contains the code to reproduce the paper `Interventional Training Reduces Causal Bypass in In-Context Introspection` on [Arxiv]() or [OpenReview](). (In submission).

<p align="center">
  <img src="./assets/teaser.png" alt="Example Box data and comical depiction of how LMs track entities" width="50%"/>
</p>

# TL;DR
Investigating whether LMs can introspect low level signals (natively or with finetuning), and if they can, how do they do it?

## Why?
Introspection research need a better evaluation task that is grounded and verifiable. Existing evaluations focuse on
either **endogenous** or **interventional** setting, never both. We argue a causally faithful introspector (vs. a causally bypassing) introspector
would have to succeed in both settings. 

Low-level signals are 
the perfect target because they are:
1. **Grounded**: They are grounded in the transformer architecture, with ground-truth labels in endogenous and interventional evaluation settings.
2. **Scalable**: You can scale this training across training data, architecture, and signal types, and trade computes for insights
3. **Locale-specific**: We know exactly which layer/position/signal index are the signals from, and can observe correlation of performance across locales.

## Findings
1. Models finetuned on output-only supervision (SFT) results in high endogenous but mostly low interventional eval, indicating causal bypass. Early layer and late token positions are easier to introspect faithfully.
2. We introduced a joint loss to fix interventional eval via interventional training. This improves interventional while maintaining endogenous performance.
3. Such training can generalize to unseen token positions, but not across layers or signal index. Each signal source has to be modified for it to be introspectable.
4. Mechanistically, attention heads responsible for either metrics overlap physically and functionally, but they are not exact.

Below are the instructions for how to use our repo.

## Signal Types
Currently we have the following types of signals written for training:
- MLP neuron (after non-linearity, post gate projection, before multiplication with up projection)
- Attention logit (pre-softmax `q.k/sqrt(d_head)`, after RoPE) at one
  `(layer, head, destination token, source token)` cell -- named `A<layer>.<head>@<src_offset>`
- Attention weight (post-softmax, code `P`): spec/parse/identity plumbing is in place,
  extraction reads the same port one step later (not wired yet)
- SAE feature

Other interesting signals one could think about extending:
- residual stream signals:
- lens results: logit lens, J-lens, etc.
- natural language autoencoders (NLA) generations

## Repo Structure
Run everything from the repo root, as `python <folder>/<script>.py`.

| Folder                  | What is in it                                                                                                                               |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `data_prep/`            | build experiment configs, cache signals into datasets, pick neurons / SAE features                                                          |
| `trainers/`             | train introspectors (SFT, with intervention, joint) and SAE detectors                                                                       |
| `evals/`                | evaluate trained introspectors: length / OOD / neuron generalization, LM degradation, benchmarks, prompted baselines                        |
| `mechanistic_analysis/` | how the introspector works: circuit tracing, head knockout / ablation, attribution, probes                                                  |
| `plotters/`             | figures and tables from finished runs                                                                                                       |
| `src/introspector/`     | shared library: configs, signal extraction, models, training utilities                                                                      |
| `scripts/`              | shell drivers for every experiment (index in `scripts/README.md`)                                                                           |
| `tests/`                | unit and smoke tests (`python -m pytest tests/`), plus model-backed sanity checks (`tests/validate_*.py`, `tests/probe_cache_alignment.py`) |
| `utils/`                | housekeeping: checkpoint cleanup, relocating runs                                                                                           |
| `outputs/plots/`        | default plotting outputs / aggregated analysis across runs                                                                                  |
| `outputs/runs/`         | individual training run caching directories                                                                                                 |

## Important Files
- `data_prep/generate_experiment_config.py`, this will take in arguments relevant to the experiment, and spit out a training json
  config, which can be consumed by trainer for reproducible experiments.
- `data_prep/cache_signals.py` this will take in the json file, and cache the signals into dataset files.
- `trainers/train_introspector.py` trains introspector with regular SFT objectives
- `trainers/train_introspector_with_intervention.py` trains introspector with intervention during training.

## Example Experiments
Checkout `.sh` files in `scripts/`; `scripts/README.md` lists what each one runs.

## Future Directions
- Change off-line training to online, refresh signal caches throughout training.
- Scale up experiments and move it to mid/pre-training, observe whether how this objective impacts representations and other task performances.
- Second order introspective tasks (i.e. "compute 1+1 but do it wrong if you detect neuron activation N2.12 > 0.5")
- Reversing the task (i.e. prompt model to change neuron activation)

## Citations
If you find our work or repo useful, please consider citing our paper!
```bibtex
@article{tang2026,
  title={Interventional Training Reduces Causal Bypass in In-Context Introspection},
  author={Tang, Zilu and Kim, Jongin and Sahin, Kerem and Atkinson, David and Mueller, Aaron and Wijaya, Derry},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```