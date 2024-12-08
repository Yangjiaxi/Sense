MODEL_PATH=/home/models/sense-sql-13b
TASK_NAME=spider-dev
OUTPUT=${MODEL_PATH}/sql-eval-results/${TASK_NAME}


python main.py \
  --model ${MODEL_PATH} \
  --tasks ${TASK_NAME} \
  --tensor_parallel_size 1 \
  --max_length_input 16384 \
  --max_new_tokens 2048 \
  --greedy \
  --do_sample False \
  --n_samples 1 \
  --save_generations_path ${OUTPUT}/generations.json \
  --eval \
  --save_result_path ${OUTPUT}/result.json