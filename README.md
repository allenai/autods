![autods_logo.png](artifacts/autods_logo.png)
# Autonomous Discovery via Surprisal

> Paper: [Open-Ended Scientific Discovery via Bayesian Surprise](https://arxiv.org/abs/2507.00310)

## Installation

Create the environment with:

```sh
conda env create -f environment.yml
```

Set environment variables:

```sh
export PYTHONPATH=$(pwd):$PYTHONPATH;
export OPENAI_API_KEY=<key>
```

## Datasets

### DiscoveryBench

```sh
git clone https://github.com/allenai/discoverybench.git temp_db
cp -r temp_db/discoverybench discoverybench
rm -rf temp_db
```

### Blade

```sh
git clone https://github.com/behavioral-data/BLADE.git temp_db
cp -r temp_db/blade_bench/datasets blade
rm -rf temp_db
```

### BYO-Datasets!
You can also use your own datasets. To do this, pass in a dataset metadata JSON file containing descriptions of the paths of datasets (relative to the metadata file) and their column descriptions in natural language. You can have a look at the metadata files in the `DiscoveryBench` directory from above as examples.

## Run AutoDS (MCTS-based hypothesis search and verification)

For example, to explore the DiscoveryBench NLS SES dataset, the following command can be used:

```sh
python src/run.py \
    --work_dir="work" \
    --out_dir="outputs" \
    --dataset_metadata="discoverybench/real/test/nls_ses/metadata_0.json" \
    --n_experiments=16 \
    --model="gpt-4o" \
    --belief_model="gpt-4o"
```

To resume a previous exploration, use the `--continue_from_dir` flag to specify the directory containing the previous
exploration logs. This will allow the script to continue from where it left off, using the MCTS nodes it had generated
so far.

## ✍️ Get in touch!

Please reach out to us on email or open a GitHub issue in case of any issues running the code: dagarwal@cs.umass.edu **(Dhruv Agarwal)**, bodhisattwam@allenai.org **(Bodhisattwa Prasad Majumder)**.

## 📄 Citation
If you find our work useful, please cite our paper:
```
@article{agarwal2025open,
  title={Open-ended Scientific Discovery via Bayesian Surprise},
  author={Agarwal, Dhruv and Majumder, Bodhisattwa Prasad and Adamson, Reece and Chakravorty, Megha and Gavireddy, Satvika Reddy and Parashar, Aditya and Surana, Harshit and Mishra, Bhavana Dalvi and McCallum, Andrew and Sabharwal, Ashish and Clark, Peter},
  journal={arXiv preprint arXiv:2507.00310},
  year={2025}
}
```
