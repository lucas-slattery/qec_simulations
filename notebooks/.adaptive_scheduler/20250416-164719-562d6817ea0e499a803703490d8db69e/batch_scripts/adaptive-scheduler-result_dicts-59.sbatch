#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --partition=obsidian-128

export EXECUTOR_TYPE=process-pool
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1



/home/lucas.slattery/micromamba/envs/system_performance/bin/python /home/lucas.slattery/micromamba/envs/system_performance/lib/python3.12/site-packages/adaptive_scheduler/_server_support/launcher.py \
    --log-fname /home/lucas.slattery/clean_repo/qec_simulations/notebooks/.adaptive_scheduler/20250416-164719-562d6817ea0e499a803703490d8db69e/logs/${NAME}-${SLURM_JOB_ID}.log \
    --job-id ${SLURM_JOB_ID} \
    --name ${NAME} \
    --url tcp://10.11.16.22:50945 \
    --log-interval 300 \
    --save-interval 300 \
    --serialized-runner-kwargs gAWVSAAAAAAAAAB9lIwEZ29hbJSMIWFkYXB0aXZlLmxlYXJuZXIuc2VxdWVuY2VfbGVhcm5lcpSMFFNlcXVlbmNlTGVhcm5lci5kb25llJOUcy4= \
    --loky-start-method loky \
    --executor-type process-pool \
    --n 1
