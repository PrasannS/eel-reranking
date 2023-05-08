# EEL: Efficiently Encoding Lattices for Reranking (preprint coming soon)
[Prasann Singhal](https://prasanns.github.io/), [Jiacheng Xu](https://jiacheng-xu.github.io/), [Xi Ye](https://www.cs.utexas.edu/~xiye/), [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/)


## Getting started

First, set up conda environment by running 

`conda create -n eelenv python=3.8`
`conda activate eelenv`
`pip install -r requirements.txt`

## Training a TFR Model

In order to run our pipeline, it's necessary to get some sort of TFR scoring model or function (we design things for transformer-based models). Credit to (https://github.com/Unbabel/COMET) which we borrow much of the structural code from. 

The model code for this can be found in the modeling_tfr folder. To train a tfr model, simply assemble a dataset (csv) with the columns src (source sequence), mt (generated sequence), and score (score based on whatever downstream metric you're using). You can then update the path to this data, and train a model with pytorch lightning by running train_tfr.sh, and keep track of the best performaing checkpoint as the TFR which you can use for EEL. 

(Note we will release our TFR models in the near future). 

# Generating lattices / experimental setup

Once we have a cTFR trained, we're ready to plug this in with a generation system to perform EEL decoding. 

For our experiments / code, we follow the pipeline of (A) Generate lattices based on decoding method (B) Perform EEL reranking, where we do so on entire sets of data. 

To generate lattices, you can navigate to the lattice-generation, and run lattice generation code. We use modified version of lattice decoding from (https://github.com/jiacheng-xu/lattice-generation). 
We include several examples of scripts that can be used to generate lattices for different settings (lattice decoding only works for specific settings, with bart-like generation models based on huggingface, you'd need to slightly adapt the code to get it to work for new base generation models / datasets). 

Once we have these, lattices can be preprocessed by calling the reverse_save_graphs function (for EEL reranking) and the explode_graphs (for baselines), from the reverse_lattice.py file, specifying the output folder from the lattice generation step. We include examples in the main function of this file, and we'll streamline this in the future. 

For our experiments, we then create score csvs based on the setting, that compute exploded TFR and downstream scorese for the generated lattices, which we use as baselines to understand the theoretical potential of our approach, though this can be ommitted in application. This is formatted in the style of the dataset, but with all possible hypotheses and their associated scores. 

# Running EEL reranking

Finally, once we have the appropriate lattice files and exploded files to compare to (optional), we can do EEL reranking. The generate_latpreds.py file includes an example of the setup for loading a TFR for a setting, and calling it on lattice files to get a dataframe with extracted ``best'' paths based on EEL reranking from TFRs (essentially storing final predictions). We'll update this to be able to run EEL with different hyperparameters with simple commands in the near future. 


More detailed instructions / adaptable code coming soon!

## Contact

prasanns@cs.utexas.edu 