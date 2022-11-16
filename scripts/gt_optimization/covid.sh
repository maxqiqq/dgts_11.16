echo ground truth meshes will be saved under ./dgts_dataset/covid/
export PYTHONPATH=$PYTHONPATH:$PWD
python process_data/ground_truth_optimization.py --tag gt --mesh-name covid --template-name sphere --num-levels 7