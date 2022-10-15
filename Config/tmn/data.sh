cd ../..
export PYTHONPATH=$PYTHONPATH:./FedWeit/
source venv/bin/activate

command="python FedWeit/main.py"
dataset="TMN"
params="-d ${dataset} --split-option non_iid --num-pick-tasks 5 --gen-num-tasks 15"
seed=42

echo ${command} -w data ${params} --random-seed ${seed}
${command} -w data ${params} --random-seed ${seed} &
