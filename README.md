# Generative modeling of living cells with implicit neural representations

<a href="mailto:wiesner@fi.muni.cz">David&nbsp;Wiesner</a>, <a href="mailto:j.m.suk@utwente.nl">Julian&nbsp;Suk</a>, <a href="mailto:s.c.dummer@utwente.nl">Sven&nbsp;Dummer</a>, <a href="mailto:xnecasovat@fi.muni.cz">Tereza&nbsp;Nečasová</a>, <a href="mailto:vladimir.ulman@vsb.cz">Vladimír&nbsp;Ulman</a>, <a href="mailto:svoboda@fi.muni.cz">David&nbsp;Svoboda</a>, and&nbsp;<a href="mailto:j.m.wolterink@utwente.nl">Jelmer&nbsp;M.&nbsp;Wolterink</a>

<b><a href="https://www.sciencedirect.com/science/article/pii/S1361841523002517" target="_blank">Journal paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://dx.doi.org/10.1007/978-3-031-16440-8_6" target="_blank">Conference paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://cbia.fi.muni.cz/files/simulations/implicit_shapes/MICCAI_2022_Oral_5_Implicit_Cell_Shapes.pdf" target="_blank">Conference slides</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://cbia.fi.muni.cz/files/simulations/implicit_shapes/MICCAI_2022_Poster_Implicit_Cell_Shapes.pdf" target="_blank">Conference poster</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://cbia.fi.muni.cz/research/simulations/implicit_shapes.html" target="_blank">Official website</a></b>




<p align="center">
<a href="https://cbia.fi.muni.cz/images/research/simulations-and-modeling/implicit_shapes/plat_new.gif" target="_blank">
    <img src="https://cbia.fi.muni.cz/images/research/simulations-and-modeling/implicit_shapes/plat_new.gif" style="margin-right: 40px;" width=30%>
</a>
<a href="https://cbia.fi.muni.cz/images/research/simulations-and-modeling/implicit_shapes/cel_new.gif" target="_blank">
    <img src="https://cbia.fi.muni.cz/images/research/simulations-and-modeling/implicit_shapes/cel_new.gif" style="margin-right: 40px;" width=30%>
</a>
<a href="https://cbia.fi.muni.cz/images/research/simulations-and-modeling/implicit_shapes/filo_new.gif" target="_blank">
    <img src="https://cbia.fi.muni.cz/images/research/simulations-and-modeling/implicit_shapes/filo_new.gif" width=30%>
</a>
</p>






## Repository structure
* ``/autodecoder`` - Autodecoder MLP for implicit representation of cell shapes.
* ``/matlab`` - Matlab scripts for data preparation and visualization.


This is the official GitHub repository of MedIA 2024 paper "Generative modeling of living cells with SO(3)-equivariant implicit neural representations" and MICCAI 2022 paper "Implicit Neural Representations for Generative Modeling of Living Cell Shapes". For more information and results, please visit our official website at <a href="https://cbia.fi.muni.cz/research/simulations/implicit_shapes" target="_blank">https://cbia.fi.muni.cz/research/simulations/implicit_shapes</a>.






## Implementation of the Method

The following guide is applicable for Linux-based systems. The versions of the libraries and command line parameters may slightly differ for Windows or macOS systems.

### Requirements and Dependencies

The implementation was tested on AMD EPYC 7713 64-Core Processor, 512 GB RAM, NVIDIA A100 80 GB GPU,
and Ubuntu 20.04 LTS with the following versions of software:

* <b>NEURAL NETWORK (``/autodecoder``)</b>
    - Python 3.9.16
    - PyTorch 2.0.1
    - PyTorch3D 0.7.4
    - NumPy 1.25.1
    - SciPy 1.11.1
    - tqdm 4.65.0
    - h5py 3.9.0
    - Spyder 5.4.3 (optional)
* <b>DATA PROCESSING AND VISUALIZATION (``/matlab``)</b>
    - Matlab R2022a
    - DIPimage 2.9 (optional)
    



### Downloads
* [Source code, pre-trained models, and examples](https://cbia.fi.muni.cz/files/simulations/implicit_shapes/impl_sdf_source_models_examples.zip) (1.2 GB)
* [Full training data set - <i>Platynereis dumerilii</i> \[HDF5\]](http://datasets.gryf.fi.muni.cz/implicit_shapes/train_plat_30x256.zip) (8.2 GB)
    * SDFs of 33 time-evolving cell shapes at 30 time points with a grid of 256&times;256&times;256.
* [Full training data set - <i>C. elegans</i> \[HDF5\]](http://datasets.gryf.fi.muni.cz/implicit_shapes/train_cele_30x256.zip) (9.1 GB)
    * SDFs of 33 time-evolving cell shapes at 30 time points with a grid of 256&times;256&times;256.
* [Full training data set - A549 filopodial \[HDF5\]](http://datasets.gryf.fi.muni.cz/implicit_shapes/train_filo_30x256.zip) (3.5 GB)
    * SDFs of 33 time-evolving cell shapes at 30 time points with a grid of 256&times;256&times;256.
* [Full training data set - Rotated <i>Platynereis dumerilii</i> \[HDF5\]](http://datasets.gryf.fi.muni.cz/implicit_shapes/train_plat_rot_8x256.zip) (6.7 GB)
    * SDFs of 100 time-evolving cell shapes at 8 time points with a grid of 256&times;256&times;256. There are 25 <i>Platynereis dumerilii</i> shapes, each randomly rotated four times over X, Y, and Z axes.
* [Full training data set - All cell types \[HDF5\]](http://datasets.gryf.fi.muni.cz/implicit_shapes/train_all_8x256.zip) (5.5 GB)
    * SDFs of 99 time-evolving cell shapes at 8 time points with a grid of 256&times;256&times;256. There are 33 <i>Platynereis dumerilii</i> shapes, 33 <i>C. elegans</i> shapes, and 33 A549 filopodial shapes.



    

### Quick Start Guide

To follow this guide, please download and extract the [Source code, pre-trained models, and examples](https://cbia.fi.muni.cz/files/simulations/implicit_shapes/impl_sdf_source_models_examples.zip) (1.2 GB) and optionally the training data sets.<br/>

* <b>Installing the Conda Environment (Optional)</b><br/>We prepared a pre-configured conda environment with all required libraries for the generative model. Conda is available <a href="https://www.anaconda.com" target="_blank">here</a>. After setting up Conda, you can install the required environment using the included <code style="color: black;font-size: 14px;">./autodecoder/conda_env.yml</code> file:<br/>
``$> conda env create -f conda_env.yml``

* <b>Shape Reconstruction</b><br/>To reconstruct the learned shape SDFs using the pre-trained models, execute the script <code style="color: black;font-size: 14px;">./autodecoder/test.py</code> with parameters specifying the desired model directory (where <code style="color: black;font-size: 14px;">plat</code> stands for <i>Platynereis dumerilii</i> cells, <code style="color: black;font-size: 14px;">cele</code> for <i>C. elegans</i> cells, and <code style="color: black;font-size: 14px;">filo</code> for filopodial cells):<br/>
``$> python test.py -x experiments/<model> -t reconstruct``<br/>
The resulting SDFs in MAT or HDF5 format will be saved in <code style="color: black;font-size: 14px;">./autodecoder/experiments/&lt;model&gt;/OUT_reconstruct</code>. You can use the Matlab script <code style="color: black;font-size: 14px;">./autodecoder/experiments/&lt;model&gt;/quick_preview.m</code> to get PNG bitmaps previewing the resulting SDFs.

* <b>Inferring New Shapes</b><br/>New SDFs are produced using randomly generated latent codes (in the case of <i>C. elegans</i> and <i>Platynereis dumerilii</i>), or by adding noise to the learned latent codes (in the case of A549 filopodial cells). To infer new SDFs using the pre-trained models, execute the script <code style="color: black;font-size: 14px;">./autodecoder/test.py</code> and specify the appropriate model directory:<br/>
``$> python test.py -x experiments/<model> -t generate``<br/>
For A549 filopodial cells, use this command:<br/>
``$> python test.py -x experiments/filo -t generate_filo``<br/>
The resulting SDFs in MAT or HDF5 format will be saved in <code style="color: black;font-size: 14px;">./autodecoder/experiments/&lt;model&gt;/OUT_randomgen</code>. You can use the Matlab script <code style="color: black;font-size: 14px;">./autodecoder/experiments/&lt;model&gt;/quick_preview.m</code> to get PNG bitmaps previewing the resulting SDFs.

* <b>Training the Network</b><br/>To train the network, download one of the training data sets and extract it to <code style="color: black;font-size: 14px;">./autodecoder/data</code> folder. The training SDFs are represented by 4D single precision floating point arrays in HDF5 format. The configuration files <code style="color: black;font-size: 14px;">./autodecoder/experiments/&lt;model&gt;/specs.json</code> contain pre-defined training parameters. To train the model, execute the <code style="color: black;font-size: 14px;">./autodecoder/train.py</code> script and specify the desired model directory:<br/>
``$> python train.py -x experiments/<model>``<br/>
Please note that the training parameters in the provided <code style="color: black;font-size: 14px;">specs.json</code> files are configured for GPUs with 80 GB of memory. To reduce the memory consumption, you can edit the configuration to reduce the number of time points in a training batch <code style="color: black;font-size: 14px;">FramesPerBatch</code> or the number of SDF points sampled per time point <code style="color: black;font-size: 14px;">TrainSampleFraction</code>. After training, you can test the resulting model using:<br/>
``$> python test.py -x experiments/<model> -t reconstruct -e <epoch>``<br/>

* <b>Preparing Your Own Training Data Sets (Matlab + CytoPacq)</b><br/> You can use 3D voxel volumes of shapes to prepare new training SDFs. An example Matlab script <code style="color: black;font-size: 14px;">./matlab/prepare_training_data/voxvol_to_sdf.m</code> uses synthetic cells generated using the CytoPacq web-interface, available at <a href="https://cbia.fi.muni.cz/simulator" target="_blank">https://cbia.fi.muni.cz/simulator</a>, to prepare training data. Basic preprocessing steps, such as shape centering and checking the number of connected components, are implemented. Supported output formats for the SDFs are MAT and HDF5. We recommend using HDF5 for larger data sets due to stronger compression and the support of data larger than 2 GB. This script expects synthetic data sets generated using CytoPacq but can be modified to suit your specific needs. Three time-evolving shapes with 30 time points generated using CytoPacq are included as an example.

* <b>Spatial and Temporal Interpolation</b><br/>The trained neural network constitutes a continuous implicit representation of the SDFs and thus is able to produce outputs in arbitrary spatial and temporal resolution. The spatial interpolation can be used to increase spatial resolution of the shapes, whereas the temporal interpolation can be used to increase the number of time points. The interpolation does not require re-training the network and can be configured by adjusting the respective parameters in <code style="color: black;font-size: 14px;">specs.json</code>. For spatial interpolation, set the parameter <code style="color: black;font-size: 14px;">ReconstructionDims</code>, and for temporal interpolation, set the parameter <code style="color: black;font-size: 14px;">ReconstructionFramesPerSequence</code>. The interpolation is applicable for reconstruction or random generation of new shapes.<br/>






## Citation

If you find our work useful in your research, please cite:

* <b>Journal paper</b>
    <div style="width: 90%; margin: auto;">
    Wiesner D, Suk J, Dummer S, Nečasová T, Ulman V, Svoboda D and Wolterink J.M. <b>Generative modeling of living cells with SO(3)-equivariant implicit neural representations</b>. Medical Image Analysis. 2024, vol. 91, p. 102991. ISSN 1361-8415. doi:<a href="https://dx.doi.org/10.1016/j.media.2023.102991" target="_blank">10.1016/j.media.2023.102991</a>.
    </div>
    
    <br/>
    <b>BibTeX</b>:
    
```
@article{wiesner2024media,
    title={Generative modeling of living cells with {SO}(3)-equivariant implicit neural representations},
    author={Wiesner, David and Suk, Julian and Dummer, Sven and Ne{\v{c}}asov{\'a}, Tereza
            and Ulman, Vladim{\'\i}r and Svoboda, David and Wolterink, Jelmer M.},
    journal={Medical Image Analysis},
    volume={91},
    pages={102991},
    year={2024},
    issn={1361-8415},
    doi={https://doi.org/10.1016/j.media.2023.102991},
    url={https://www.sciencedirect.com/science/article/pii/S1361841523002517}
}
```

* <b>Conference paper</b>
    <div style="width: 90%; margin: auto;">
    Wiesner D, Suk J, Dummer S, Svoboda D and Wolterink J.M. <b>Implicit Neural Representations for Generative Modeling of Living Cell Shapes</b>. In Linwei Wang, Qi Dou, P. Thomas Fletcher, Stefanie Speidel, Shuo Li. International Conference on Medical Image Computing and Computer Assisted Intervention. Switzerland: Springer Nature Switzerland, 2022. p. 58-67., ISBN 978-3-031-16440-8. doi:<a href="https://dx.doi.org/10.1007/978-3-031-16440-8_6" target="_blank">10.1007/978-3-031-16440-8_6</a>. 
    </div>
    
    <br/>
    <b>BibTeX</b>:
    
```
@InProceedings{wiesner2022miccai,
    title={Implicit Neural Representations for Generative Modeling of Living Cell Shapes},
    author={Wiesner, David and Suk, Julian and Dummer, Sven and Svoboda, David and Wolterink, Jelmer M.},
    editor={Wang, Linwei and Dou, Qi and Fletcher, P. Thomas and Speidel, Stefanie and Li, Shuo},
    booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
    year={2022},
    publisher={Springer Nature Switzerland},
    address={Cham},
    pages={58--67},
    isbn={978-3-031-16440-8},
    doi={10.1007/978-3-031-16440-8_6}
}
```







## Acknowledgements

This work was partially funded by the 4TU Precision Medicine programme supported by High Tech for a Sustainable Future, a framework commissioned by the four Universities of Technology of the Netherlands. Jelmer M. Wolterink was supported by the NWO domain Applied and Engineering Sciences VENI grant (18192). We acknowledge the support by the Ministry of Education, Youth and Sports of the Czech Republic (MEYS CR) (Czech-BioImaging Projects LM2023050 and CZ.02.1.01/0.0/0.0/18_046/0016045). This project has received funding from the European High-Performance Computing Joint Undertaking (JU) and from BMBF/DLR under grant agreement No 955811. The JU receives support from the European Union’s Horizon 2020 research and innovation programme and France, the Czech Republic, Germany, Ireland, Sweden and the United Kingdom.

The data set of <i>Platynereis dumerilii</i> embryo cells is courtesy of <a href="https://linkedin.com/in/mette-handberg-thorsager" target="_blank">Mette Handberg-Thorsager</a> and <a href="https://scholar.google.com/citations?user=pexj_eQAAAAJ" target="_blank">Manan Lalit</a>, who both have kindly shared it with us.

The shape descriptors in the paper were computed and plotted using an online tool for quantitative evaluation, Compyda, available at <a href="https://cbia.fi.muni.cz/compyda" target="_blank">https://cbia.fi.muni.cz/compyda</a>. We thank its authors Tereza Nečasová and Daniel Múčka for kindly giving us early access to this tool and facilitating the evaluation of the proposed method.

The neural network implementation is based on <a href="https://github.com/facebookresearch/DeepSDF" target="_blank">DeepSDF</a>, <a href="https://github.com/cvlab-epfl/MeshSDF" target="_blank">MeshSDF</a>, and <a href="https://github.com/vsitzmann/siren" target="_blank">SIREN</a>.
