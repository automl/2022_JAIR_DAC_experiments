## Learning Step-Size Adaptation in CMA-ES

Learning step-size adaptation policies as in [(Shala et al, 2016)](https://www.microsoft.com/en-us/research/publication/learning-step-size-controllers-for-robust-neural-network-training/), but using Sequential Model-based Algorithm Configuration [(SMAC3)](https://arxiv.org/abs/2109.09831) instead of Relative Entropy Policy Search (REPS).

#### Installation instructions:

Create and activate conda environment:
```
conda create --name dac_sgd python=3.6
conda activate dac_cma
```
Install DACBench
```
git clone https://github.com/automl/DACBench.git
cd DACBench
pip install -e .[example]
```

Install (remaining) project requirements:
```
pip install -r requirements.txt
```
## Experiment Setup

### Plotting the data
The plotting script can be found in the folder *scripts*. The data from the trained policy is in *examples/DAC_Journal* whereas for CSA it is in *data/CSA_10D*.
You can plot the results by running the following command (e.g. for GallaghersGaussian21hi):
```
python plot_performance.py --lto_path examples/DAC_Journal/GallaghersGaussian21hi_LTO.json --csa_path data/CSA_10D/GallaghersGaussian21hi.json --function GallaghersGaussian21hi
```
### Training
```
python gps_train.py DAC_Journal
```
- The output of training is the pickled version of the learned policy, saved in the path *DAC_Journal/data_files*.
### Testing
```
python gps_test.py DAC_Journal
```