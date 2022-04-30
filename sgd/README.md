## Learning Rate Control in Neural Network Training

Meta-learning learning rate control policies as in [(Daniel et al, 2016)](https://www.microsoft.com/en-us/research/publication/learning-step-size-controllers-for-robust-neural-network-training/), but using Sequential Model-based Algorithm Configuration [(SMAC3)](https://arxiv.org/abs/2109.09831) instead of Relative Entropy Policy Search (REPS).

#### Installation instructions:
(_using conda, python 3.6_)

Create and activate conda environment:
```
conda create --name dac_sgd python=3.6
conda activate dac_sgd
```
Install DACBench
```
git clone https://github.com/automl/DACBench.git
cd DACBench
git submodule update --init --recursive
pip install -e .[example]
cd ..
```

Install (remaining) project requirements:
```
pip install -r requirements.txt
```

#### Meta-training:
To meta-train a learning rate controller for RMSprop using the setup in the paper:
```
python run_meta_training.py dac/setups/dac_rmsprop.json <seed>
```
and 
```
python run_meta_training.py dac/setups/dac_momentum.json <seed>
```
to meta-train a learning rate controller for Momentum.

Replace ```<seed>``` by an integer (used by SMAC to seed its random number generator)

#### Meta-testing learning rate control policies:
To test a learned learning rate controller:
```
python run_meta_test.py --optimizer <optimizer> --dataset <dataset> --policy <policy>
```
Replace ```<optimizer>``` by ```"rmsprop"``` or ```"momentum"```, ```<dataset>``` by ```"mnist"``` or ```"cifar"```, and ```<policy>``` to a path containing a configuration stored by the meta-training process (.npy file). For convenience, the final policies obtained by the 10 meta-training runs discussed in the paper are provided in ```dac/learned_policies```.

To run one of the constant learning rate baselines:
```
python run_meta_test.py --optimizer <optimizer> --dataset <dataset> --baseline <lr>
```

Replace ```<lr>``` with the learning rate (e.g., 0.001)
