# NeuropixelsRegistration
 Motion estimation and registration of Neuropixels data (python).
 
 We provide the user with two modules: registration module (estimate_displacement.py) and preprocessing module (preprocess.py).
 
 4d_register.ipynb provides a Jupyter Notebook example of how to register and preprocess raw int16 .bin file and output registered/preprocessed float32 .bin file.
 
 To use yass, follow the installation instruction at https://github.com/paninski-lab/yass/tree/master.
 To use the yass READER, change the drift.yaml file. You need to change: 
 root_folder, recordings, geometry under data, and n_channels under recordings.