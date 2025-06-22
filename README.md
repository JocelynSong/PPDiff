<h1>PPDiff: Diffusing in Hybrid Sequence-Structure Space for Protein-Protein Complex Design</h1>

<h2>Model Architecture</h2>

This repository contains code, data and model weights for ICML 2025 paper [PPDiff: Diffusing in Hybrid Sequence-Structure Space for Protein-Protein
Complex Design](https://arxiv.org/pdf/2506.11420)

The overall model architecture is shown below:

![image](./ProComplexDiff.png)


<h2>Environment</h2>
The dependencies can be set up using the following commands:

```ruby
conda env create -f PPDiff.yml 
conda activate PPDiff 
bash setup.sh 
```

<h2>Download Data</h2>

We provide our curated protein-protein complex dataset PPBench at [PPBench](https://drive.google.com/file/d/1DmvVKvZIVxT4-bxIIZ4bRQtIrQJ2QwJN/view?usp=sharing) 

Please download the dataset and put them in the data folder.

```angular2html
mkdir data 
cd data 
wget https://drive.google.com/file/d/1DmvVKvZIVxT4-bxIIZ4bRQtIrQJ2QwJN/view?usp=drive_link
```

<h2>Download Model</h2>

We provide the checkpoint of general protein-protein complex design task used in the paper at [Model](https://drive.google.com/file/d/19SmgY7sXIPN2Wk5Rln7x9jzzJ5np-Hmj/view?usp=sharing) 


Please download the checkpoints and put them in the models folder.

If you want to train your own model, please follow the training guidance below

<h2>Training</h2>
If you want to train a model from scratch, please follow the script below:

```ruby
bash train_complex_data_diffusion.sh
```

<h2>Inference</h2>
To design general protein-protein complexes, please use the following scripts:

```ruby
bash generation.sh
```

There are three items in the output directory:

1. target.txt refers to the target protein sequences
2. binder.true.txt refers to the input binder sequences
3. binder.gen.txt refers to the designed binder sequences


<h2>Citation</h2>
If you find our work helpful, please consider citing our paper.

```
@article{song2025ppdiff,
  title={PPDiff: Diffusing in Hybrid Sequence-Structure Space for Protein-Protein Complex Design},
  author={Song, Zhenqiao and Li, Tiaoxiao and Li, Lei and Min, Martin Renqiang},
  journal={arXiv preprint arXiv:2506.11420},
  year={2025}
}
```
