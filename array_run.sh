#! /bin/bash

# If SGE_TASK_ID is N, set SEQ_FILE to the Nth line of "sequence-files"

task_id=$SLURM_ARRAY_TASK_ID
echo $task_id

# pulls out the $task_id line from the joblist, which should be a string defining the parameters passed to the job
SEQ_FILE=`awk 'NR=='$task_id'{print}' /home/ec2-user/Mutator/joblist.txt`
echo $SEQ_FILE

python3 mutator_simulation_with_variable_effect_size.py $SEQ_FILE
