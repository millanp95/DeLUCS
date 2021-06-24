# DeLUCS
This repository contains all the source files required to run the DeLUCS algorithm for DNA sequences, as well as a detailed guide for running the code .


## Computational Pipeline: 


### 1. Build the dataset:
  ```
  	python build_dp.py --data_path=<PATH_sequence_folder>	
  ```
 * Input: Folders with the sequences in FASTA format
 * Output : file in the form (label,sequence,accession)


### 2. Compute the pairs.

  ```
	python get_pairs.py --data_path=<PATH_pickle_dataset> --k=6 --modify='mutation' --output=<PATH_output_file>
  ```
* Input: file in the form (label,sequence,accession)
* Output : file in the form of (pairs, x_test, y_test)
  
### 3. Train the model.

* For training DeLUCS to cluster your own data: 
	```
	python TrainDeLUCS.py --data_dir=<PATH_of_computed_mimics> --out_dir=<OUTPURDIR>
	```
 	* Input: Pickle file with the mimics. 
   	* Output : Pickle file with the cluster assignments for each sequence. 
		
		
* For testing the performance of DeLUCS with your own data (Labels Available)
	```
	python EvaluateDeLUCS.py --data_dir=<PATH_of_computed_mimics> --out_dir=<OUTPURDIR>
	```

	* Input: Pickle file with the mimics in the form of (pairs, x_test, y_test). 
	* Output : 
			* Image with the confusion Matrix. 
			* File with the misclassified sequences in the form (accession, true_label, predicted_label)

* For training a single Neural Networks in an unsupervised way:
	```
	python SingleRun.py --data_dir=<PATH_of_computed_mimics> --out_dir=<OUTPURDIR>
	```
	```
	python EvaluateSingleRun.py --data_dir=<PATH_of_computed_mimics> --out_dir=<OUTPURDIR>
	```



		
	

<!--in one of the Compute Canada clusters available for our lab.

<!-- ## Accesing the resources:

<!-- In our lab we have acces to three different clusters within the Compute Canada infraestructure: Cedar and Graham. We can acces the cluster via ssh using the Compute Canada credentials and the name of the cluster we want to access:
```
ssh pmillana@cedar.computecanada.ca
ssh pmillana@graham.computecanada.ca
```
## Different File Systems: 
Once you have accessed the cluster trough a login node (Do not run anything on this nodes), you will see that all our folders are under the account ```def-khill22```, this is the account name of our group and should be used for every job submition. 
For each user in our account there are differnt file systems that should be used for different purposes: 
<!-- 
* **HOME**: While your home directory may seem like the logical place to store all your files and do all your work, in general this isn't the case - your home normally has a relatively small quota and doesn't have especially good performance for the writing and reading of large amounts of data. The most logical use of your home directory is typically source code, small parameter files and job submission scripts.
* **PROJECT**: The project space has a significantly larger quota and is well-adapted to sharing data among members of a research group since it, unlike the home or scratch, is linked to a professor's account rather than an individual user.
* **SCRATCH**: For intensive read/write operations, scratch is the best choice. Remember however that important files must be copied off scratch since they are not backed up there, and older files are subject to purging. The scratch storage should therefore only be used for transient files.
<!-- 
<p align="center">
  <img src ="Images\Screenshot from 2020-06-02 19-41-06.png" alt="drawing" width="500"/>
</p>

<!-- 
The following table is taken from the Compute Canada documentation and show all the policies for each file system:
<!-- 
<p align="center">
  <img src ="Images\Screenshot from 2020-06-02 19-41-15.png" alt="drawing" width="500"/>
</p>



<!-- 
For transfering local files to the cluster you can use ```scp``` with the same credentials you used for logging into the system: 

```
 scp  path_to_local_files  pmillana@cedar.computecanada.ca:~/desired_folder_inside_home_directory
```
<!-- 
For more information see: https://docs.computecanada.ca/wiki/Storage_and_file_management and https://docs.computecanada.ca/wiki/Storage_and_file_management#Filesystem_quotas_and_policies

<!-- 
## Sumbitting Jobs: 
Compute Canada uses SLURM https://slurm.schedmd.com/documentation.html for managing jobs and allocating resources within the different clusters. To submit a job you will need to create a sbatch script with all the requirements that are neccessary for running your code. 
<!-- 
**Note**: Submitting jobs from directories residing in /home is not permitted, transfer the sbatch script either to your /project or /scratch directory and submit the job from there.
<!-- 
You can also run your code inside an interactive node, this is recommended before submitting bigger jobs, an example of that can be:

 ``` (bash)
  salloc --account=def-khill22 --gres=gpu:1 --cpus-per-task=4 --mem=32000M --time=0-00:25:00
 ```
 <!-- 
 You can run separately the commands in the following sbatch script inside your interactive node with few iterations to check that the code doesn't have any error. This is an example of the  script that was used in our case: 
 ```
#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-02:00     # DD-HH:MM:SS
<!-- 
module load python/3.6 cuda cudnn
<!-- 
SOURCEDIR=~/src   #I copied the files in this directory inside my home directory
<!-- 
# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt
<!-- 
# Start training
python TrainCluster.py --data_path ~/scratch/data/train.p --load_data True
# --data_path: Path of the decompressed training data.
# --load_features: True if the training features are precomputed.
 ```
 For running the script you run: 
 ```
 sbatch --account=def-khill22 script.sh
 ```
 For monitoring the status of your job you can run: 
 
```
squeue --account=def-khill22  
```
<!-- 
A log file with the output of your job will be created after it finishes in the same directory of your sbash sript. 

 

