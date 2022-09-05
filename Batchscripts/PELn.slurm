#!/bin/bash

#SBATCH --job-name=PELn
#SBATCH --partition=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gubaidulinvadim@gmail.com
#SBATCH --time=08:00:00
#SBATCH --array=0-650
#SBATCH --export=ALL
#SBATCH --error=/lustre/bhs/vgubaid/err/job%J.err
#SBATCH --output=/lustre/bhs/vgubaid/out/job%J.out
source /cvmfs/aph.gsi.de/modules.sh
module load python3.7 anaconda3
export PYTHONPATH='/lustre/bhs/vgubaid/PyHEADTAIL/'
export HDF5_USE_FILE_LOCKING='FALSE'

FILE='/lustre/bhs/vgubaid/Simulation/LHC-Antidamper/PELntrack.py'
FOLDER='/lustre/bhs/vgubaid/Data/LHC/antidamper/PELn/'
TMP_FOLDER='/tmp/vgubaid/antidamperPELn/'
rm -rf ${TMP_FOLDER}
mkdir -p ${TMP_FOLDER}
mkdir -p ${FOLDER}
CHROMATICITY_0=0.
dQmax=0.001
RATIO=1.0
python ${FILE} ${SLURM_ARRAY_TASK_ID} ${CHROMATICITY_0} ${dQmax} ${TMP_FOLDER} ${RATIO}
rsync -avP ${TMP_FOLDER} ${FOLDER}
# rm /lustre/bhs/vgubaid/out/*
# rm /lustre/bhs/vgubaid/err/*