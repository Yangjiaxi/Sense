MODEL_PATH=/home/models/sense-sql-13b
TASK_NAME=spider-train
OUTPUT=${MODEL_PATH}/sql-eval-results/${TASK_NAME}

# To debug, set --limit N to run up to N questions (e.g., --limit 10).
# For evaluating the full dataset, omit the --limit flag.

python main.py \
  --model ${MODEL_PATH} \
  --tasks ${TASK_NAME} \
  --tensor_parallel_size 8 \
  --max_length_input 16384 \
  --max_new_tokens 2048 \
  --do_sample True \
  --temperature 0.8 \
  --n_samples 104 \
  --limit 10 \
  --save_generations_path ${OUTPUT}/generations.json \
  --eval \
  --save_result_path ${OUTPUT}/result.json

