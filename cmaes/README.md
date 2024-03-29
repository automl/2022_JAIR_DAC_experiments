## Learning Step-Size Adaptation in CMA-ES

Learning step-size adaptation policies as in [(Shala et al, 2020)](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-PPSN-LTO-CMA.pdf).

#### Installation instructions:

Create and activate conda environment:
```
conda create --name dac_cma python=3.6
conda activate dac_cma
```
Install DACBench
```
git clone https://github.com/automl/DACBench.git
cd DACBench
git checkout v0.1
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
python source/gps/gps_train.py DAC_Journal
```
- The output of training is the pickled version of the learned policy, saved in the path *DAC_Journal/data_files*.
### Testing
```
python source/gps/gps_test.py DAC_Journal
```
