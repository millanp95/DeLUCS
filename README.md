# DeepCluster
This repository contains all the source files required to run the Deep clustering algorithm for DNA sequences, as well as a detailed guide for running the code in one of the Compute Canada clusters available for our lab.

## Accesing the resources:

In our lab we have acces to three different clusters within the Compute Canada infraestructure: Cedar and Graham. We can acces the cluster via ssh using the Compute Canada credentials and the name of the cluster we want to access:
```
ssh pmillana@cedar.computecanada.ca
ssh pmillana@graham.computecanada.ca
```
## Different File Systems: 
Once you have accessed the cluster trough a login node (Do not run anything on this nodes), you will see that all our folders are under the account ```def-khill22```, this is the account name of our group and should be used for every job submition. 
For each users in our account there are differnt file systems that should be used for different purposes: 

The following table is taken from the Compute Canada documentation and show all the policies for each file system:

For transfering local files to the cluster you can use ```scp``` with the same credentials you used for logging into the system: 

```
 scp  path_to_local_files  pmillana@cedar.computecanada.ca:~/desired_folder_inside_home_directory
```



## Sumbitting Jobs: 
Compute Canada uses SLURM https://slurm.schedmd.com/documentation.html for managing jobs and allocating resources within the different clusters. To submit a job you will need to create a sbatch script with all the requirements that area neccessary for running your code. You can also run your code inside an interactive node, this is recommended before submitting bigger jobs. 


$Note$:Submitting jobs from directories residing in /home is not permitted, transfer the files either to your /project or /scratch directory and submit the job from there.


Creating an interactive job for testing the code after submitting the cod via slurm. 
  salloc --account=def-khill22 --gres=gpu:1 --cpus-per-task=4 --mem=32000M --time=0-00:25:00
  ```
 
