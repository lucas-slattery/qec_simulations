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
    --log-fname /home/lucas.slattery/clean_repo/qec_simulations/notebooks/.adaptive_scheduler/20250415-211745-6d733b2da45a42119838250dec868771/logs/${NAME}-${SLURM_JOB_ID}.log \
    --job-id ${SLURM_JOB_ID} \
    --name ${NAME} \
    --url tcp://10.11.16.22:41109 \
    --log-interval 300 \
    --save-interval 300 \
    --serialized-runner-kwargs gAWVSAAAAAAAAAB9lIwEZ29hbJSMIWFkYXB0aXZlLmxlYXJuZXIuc2VxdWVuY2VfbGVhcm5lcpSMFFNlcXVlbmNlTGVhcm5lci5kb25llJOUcy4= \
    --n 1 \
    --executor-type process-pool \
    --loky-start-method loky
