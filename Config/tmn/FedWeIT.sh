cd ../..
export PYTHONPATH=$PYTHONPATH:./FedWeit/
source venv/bin/activate

command="python FedWeit/main.py"
dataset="TMN"
port_num=11100

base_params="-d ${dataset} --split-option non_iid --num-pick-tasks 5 --gen-num-tasks 15 --fed-method 0"
et_params=""
additional_params="--exhaust-tasks"

other_params_all=("" "--fedweit-dense")
lambdas=("1" "0.1")
seeds=("1" "2")

for seed in "${seeds[@]}" ; do
  for lambda in "${lambdas[@]}" ; do
    for other_params in "${other_params_all[@]}" ; do
      params="--random-seed ${seed} --random-seed-task-alloc ${seed} --host-port ${port_num} "
      params="${params} ${other_params} --lambda-two ${lambda}"
      params="${params} ${et_params} ${additional_params} ${base_params}"
      echo ${command} ${params}

      ${command} -w server ${params} &
      sleep 5
      ${command} -w client ${params} &
      sleep 55
      ((port_num++))
      done
    done
  done
