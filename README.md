# Generative modeling of living cells with implicit neural representations

<a href="mailto:wiesner@fi.muni.cz">David&nbsp;Wiesner</a>, <a href="mailto:j.m.suk@utwente.nl">Julian&nbsp;Suk</a>, <a href="mailto:s.c.dummer@utwente.nl">Sven&nbsp;Dummer</a>, <a href="mailto:xnecasovat@fi.muni.cz">Tereza&nbsp;Nečasová</a>, <a href="mailto:vladimir.ulman@vsb.cz">Vladimír&nbsp;Ulman</a>,<br/><a href="mailto:svoboda@fi.muni.cz">David&nbsp;Svoboda</a>, and&nbsp;<a href="mailto:j.m.wolterink@utwente.nl">Jelmer&nbsp;M.&nbsp;Wolterink</a>

<b><a href="https://arxiv.org/abs/2304.08960" target="_blank">Journal paper (arXiv)</a>&nbsp;&nbsp;|&nbsp;&nbsp;Conference paper (<a href="https://dx.doi.org/10.1007/978-3-031-16440-8_6" target="_blank">Springer</a>/<a href="https://arxiv.org/abs/2207.06283" target="_blank">arXiv</a>)&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://cbia.fi.muni.cz/files/simulations/implicit_shapes/MICCAI_2022_Oral_5_Implicit_Cell_Shapes.pdf" target="_blank">Slides</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://cbia.fi.muni.cz/files/simulations/implicit_shapes/MICCAI_2022_Poster_Implicit_Cell_Shapes.pdf" target="_blank">Poster</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://cbia.fi.muni.cz/research/simulations/implicit_shapes.html" target="_blank">Official website</a></b>



## Repository structure
* ``/archive`` - Archived source codes.
* ``/autodecoder`` - Autodecoder MLP for implicit representation of cell shapes.
* ``/matlab`` - Matlab scripts for data preparation and visualization.



## Implementation of the Method

### Requirements and Dependencies
The implementation was tested on AMD EPYC 7713 64-Core Processor, 512 GB RAM, NVIDIA A100 80 GB GPU, and Ubuntu 20.04 LTS with the following versions of software:

* <b>NEURAL NETWORK (``/autodecoder``)</b>
    - Python 3.10.10
    - NumPy 1.24.3
    - SciPy 1.10.1
    - PyTorch 2.0.0
    - PyTorch3D 0.7.3
    
* <b>DATA PROCESSING AND VISUALIZATION (``/matlab``)</b>
    - Matlab R2021a
    - DIPimage 2.9 (optional)
    
### Download
* [Source Code, Pre-Trained Models, and Example Datasets \[ZIP\]](https://cbia.fi.muni.cz/files/simulations/implicit_shapes/impl_sdf_source.zip) (1.5 GB)
* [Full Training Dataset \[ZIP\]](https://cbia.fi.muni.cz/simulator/data/train_sdfs.zip) (24 GB)
    
### Quick Start

* <b>Installing the Anaconda Environment (Optional)</b><br/>We prepared a pre-configured Anaconda environment with all required libraries for the generative model. Anaconda is available <a href="https://www.anaconda.com" target="_blank">here</a>. After setting up Anaconda and extracting the source code ZIP archive, you can install the required environment using the included <code style="color: black;font-size: 14px;">env_impl_sdf.yml</code> file:<br/>
``$> conda env create -f env_impl_sdf.yml``

* <b>Shape Reconstruction</b><br/>To reconstruct the learned SDFs using the pre-trained models, execute the script <code style="color: black;font-size: 14px;">reconstruct.py</code> with parameters specifying the desired model directory (<code style="color: black;font-size: 14px;">filo</code> stands for filopodial cells, <code style="color: black;font-size: 14px;">cele</code> for <i>C. elegans</i> cells):<br/>
``$> python reconstruct.py -e experiments/filo_cells_pretrained``<br/>
``$> python reconstruct.py -e experiments/cele_cells_pretrained``<br/>
The resulting SDFs in MAT format will be saved in <code style="color: black;font-size: 14px;">./experiments/&lt;model&gt;/reconstructions</code>.

* <b>Inferring New Shapes</b><br/>New SDFs are produced using randomly generated latent codes (in the case of <i>C. elegans</i>), or by interpolating between the learned latent codes (in the case of filopodial cells). To infer new SDFs using the pre-trained model, execute the script <code style="color: black;font-size: 14px;">infer_new_filo.py</code> or <code style="color: black;font-size: 14px;">infer_new_cele.py</code>, and specify the appropriate model directory:<br/>
``$> python infer_new_filo.py -e experiments/filo_cells_pretrained``<br/>
``$> python infer_new_cele.py -e experiments/cele_cells_pretrained``<br/>
The resulting SDFs in MAT format will be saved in <code style="color: black;font-size: 14px;">./experiments/&lt;model&gt;/interpolations</code>.

* <b>Training the network</b><br/>The source archive contains example training datasets for both cell classes. The training SDFs are in the <code style="color: black;font-size: 14px;">./data/&lt;cell_class&gt;</code> directory and are saved as 4D floating point arrays in MAT format. The configuration file <code style="color: black;font-size: 14px;">specs.json</code> with pre-defined training parameters is in the respective <code style="color: black;font-size: 14px;">./experiments/&lt;model&gt;</code> directory. To train the model, execute the <code style="color: black;font-size: 14px;">train.py</code> script and specify the desired model directory:<br/>
``$> python train.py -e experiments/filo_cells_demo``<br/>
``$> python train.py -e experiments/cele_cells_demo``

* <b>Spatial and temporal interpolation</b><br/>A trained neural network is a continuous implicit representation of the SDFs and thus is able to produce outputs in arbitrary spatial and temporal resolution. The spatial interpolation can be used to increase spatial resolution of the shapes, whereas the temporal interpolation can be used to increase the number of captured time points. The interpolation does not require re-training the network and can be configured by adjusting the respective parameters in <code style="color: black;font-size: 14px;">specs.json</code>. For spatial interpolation, set the parameter <code style="color: black;font-size: 14px;">ReconstructionDims</code>, and for temporal interpolation, set the parameter <code style="color: black;font-size: 14px;">ReconstructionFramesPerSequence</code>. The interpolation is applicable for reconstruction or inference of new shapes:<br/>
``$> python reconstruct.py -e experiments/<model>``<br/>
``$> python infer_new_<cell_class>.py -e experiments/<model>``<br/>   

* <b>Converting SDFs to voxel volumes (Matlab)</b><br/>To convert the SDFs to a sequence of 3D TIFF images, use the respective script <code style="color: black;font-size: 14px;">./matlab_scripts/convert_sdf_to_tif_volume&#95;&lt;cell_class&gt;.m</code>. The example output of the script for the included training datasets is in the <code style="color: black;font-size: 14px;">./matlab/volume&#95;&lt;cell_class&gt;_demo_01</code> directory.

* <b>Visualization of SDFs (Matlab)</b><br/>To render visualizations of the SDFs, use the included script <code style="color: black;font-size: 14px;">./matlab_scripts/render_sdf&#95;&lt;cell_class&gt;.m</code>. The example output for the included datasets is in the <code style="color: black;font-size: 14px;">./matlab/renders&#95;&lt;cell_class&gt;_demo_01</code> directory.

### Citation
<div style="width: 90%; margin: auto;">
Wiesner D, Suk J, Dummer S, Svoboda D a Wolterink J. <b>Implicit Neural Representations for Generative Modeling of Living Cell Shapes</b>. In Linwei Wang, Qi Dou, P. Thomas Fletcher, Stefanie Speidel, Shuo Li. International Conference on Medical Image Computing and Computer Assisted Intervention. Switzerland: Springer Nature Switzerland, 2022. p. 58-67., ISBN 978-3-031-16440-8. doi:<a href="https://dx.doi.org/10.1007/978-3-031-16440-8_6" target="_blank">10.1007/978-3-031-16440-8_6</a>. 
</div>

### Acknowledgements
This work was partially funded by the 4TU Precision Medicine programme supported by High Tech for a Sustainable Future, a framework commissioned by the four Universities of Technology of the Netherlands. Jelmer M. Wolterink was supported by the NWO domain Applied and Engineering Sciences VENI grant (18192). David Wiesner was supported by the Grant Agency of Masaryk University under the grant number MUNI/G/1446/2018. David Svoboda was supported by the MEYS CR (Projects LM2018129 and CZ.02.1.01/0.0/0.0/18_046/0016045 Czech-BioImaging). 



## Baseline models

### Dependencies & packages
* Python (tested on 3.8.6)
* PyTorch (tested on 1.9.0)
* PyTorch Geometric (tested on 1.7.2) with
  * torch-cluster (tested on 1.5.9)
  * torch-scatter (tested on 2.0.8)
  * torch-sparse (tested on 0.6.11)
  
### Installing Anaconda environments
```
conda env create -f mesh-sdf.yml
conda env create -f siren-cuda10.yml
```

### Running MeshSDF
```
python train_deep_sdf.py -e experiments/cells
```

### Running Siren
```
python train_video.py --model_type=sine --experiment_name cat_video --dataset cat

python test_video.py --checkpoint_path=./logs/cat_video/checkpoints/model_current.pth --experiment_name=cat_video --dataset cat
```
