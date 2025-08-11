# DeepLearning-LatentSpace-StabilityEvaluation

   Mabadeje, A. O., Morales, M. M., Torres-Verdin, C., and Pyrcz, M. (2025). Evaluating the stability of deep learning latent feature space for subsurface modeling. *Mathematical Geosciences*. https://doi.org/10.1007/s11004-025-10223-3

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-summary"> ➤ Project Summary</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#acknowledgements"> ➤ Acknowledgements</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PROJECT SUMMARY -->
<h2 id="project-sumary"> :pencil: Project Summary</h2>

<p align="justify"> 
 
* Current nonlinear dimensionality reduction methods are based on various assumptions yielding non-unique solutions that increase uncertainty and reduce reliability. 

* We propose a generalizable workflow to evaluate the stability of deep learning latent feature spaces to aid model reliability when performing predictive, and inferential analyses. 


* Workflow is demonstrated on 2 end-member cases and a publicly available dataset, promoting improved model interpretability, reliability, and quality control for more informed decision-making for diverse analytical workflows.


* We recommend evaluating such spaces before making inferences and subsequent decisions to reduce algorithmic bias.
  
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

<!--This project is written in Python programming language. <br>-->
The following open-source packages are mainly used in this project:
* Numpy
* Pandas
* Matplotlib
* Scikit-Learn
* Pytorch

Please install other required packages detailed in the `requirements.txt` file and include custom-made `Functions_comparisons.py` containing functions in the active working directory

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    Functions
    .
    ├── Functions.py

    Scenarios
    .
    ├── Notebook 0: EigenPCA Data Generation
    ├── Notebook 1: Correlation 0
    ├── Notebook 2: Public Dataset
    ├── Notebook 3: Correlation 1

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  The dataset used for this demonstration is publicly available in <a href="[https://github.com/GeostatsGuy](https://github.com/GeostatsGuy/GeoDataSets/blob/master/unconv_MV_v4.csv)"> GeoDataSets: Synthetic Subsurface Data Repository as `unconv_MV_v4.csv` </a> 
  
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ACKNOWLEDGEMENTS -->
<h2 id="acknowledgements"> :scroll: Acknowledgements</h2>
<p align="justify"> 
This work is supported by the Digital Reservoir Characterization Technology (DIRECT) Industry Affiliate Program at the University of Texas at Austin, and many thanks to Equinor for providing real dataset used to test the workflow.
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> :scroll: Contributors</h2>

<p>  
  👩‍🎓: <b>Ademide O. Mabadeje</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>ademidemabadeje[at]utexas[dot]edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/Mide478">@Mide478</a> <br>

  👩‍🎓: <b>Misael M. Morales</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>misaelmorales[at]utexas[dot]edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/misaelmmorales">@misaelmmorales</a> <br>

  👩‍🎓: <b>Carloss Torres-Verdin</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>cverdin[at]austin[dot]utexas[dot]edu</a> <br>
  
  👨‍🏫: <b>Michael J. Pyrcz</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>mpyrcz[at]austin[dot]utexas[dot]edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/GeostatsGuy">@GeostatsGuy</a> <br>
</p>
<br>
