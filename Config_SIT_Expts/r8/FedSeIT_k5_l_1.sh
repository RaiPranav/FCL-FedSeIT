cd ../..
export PYTHONPATH=$PYTHONPATH:./FedWeit/
source venv/bin/activate

command="python FedWeit/main.py"
dataset="reuters8"
port_num=15200
top_k="5"

base_params="-d ${dataset} --split-option non_iid --num-pick-tasks 5 --gen-num-tasks 15 --fed-method 0"
et_params="--embedding-transfer --et-top-k ${top_k} --et-use-clusters --et-init-alphas --et-alphas-trainable --et-task-similarity-method rectified_linear_normalised"
additional_params="--exhaust-tasks --model 5 --concatenate-aw-kbs"

other_params_all=("" "--dense-detached" "--dense-detached --project-adaptives")
lambdas=("1")
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
