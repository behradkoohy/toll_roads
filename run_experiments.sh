#!/bin/bash

# Get the number of folders to create from the command line argument
N=$1

# Create a directory for the experiment
date_str=$(date +%d%m%Y)_$2_$3
experiment_dir="experiment_${date_str}_highway_50"
mkdir /scratch/bk2g18/${experiment_dir}

# Create N folders inside the experiment directory
for i in $(seq 0 ${N})
do
  folder_name="exp_${i}"
  mkdir "/scratch/bk2g18/${experiment_dir}/${folder_name}"

  # Run the Experiments
  sbatch create_job.slurm 300 200 10000 /scratch/bk2g18/${experiment_dir}/${folder_name}/ $2 $3
done

# Create a cleanup script
cat <<EOF > cleanup.sh
#!/bin/bash

# Remove the experiment directory
rm -rf /scratch/bk2g18/${experiment_dir}
rm *.out
EOF

# Make the cleanup script executable
chmod +x cleanup.sh 
