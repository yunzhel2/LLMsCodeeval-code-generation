### code_review_smell_testing

Steps

Replace "openai_api_key" with your personal OpenAI API key.

1. `cd code-data-eval`
2. `pip install -r requirements.txt`
3. `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-3.5-turbo --data_load_name code_review_data.jsonl --result_save_name code_review_eval_gpt3.jsonl --log_file_name code_review_eval_gpt3.log`
4. `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-3.5-turbo --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_gpt3.jsonl --log_file_name code_smell_eval_gpt3.log`
5. `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-3.5-turbo --data_load_name code_test_data.jsonl --result_save_name code_test_data_gpt3.jsonl --log_file_name code_test_data_gpt3.log`
6. `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-4 --data_load_name code_review_data.jsonl --result_save_name code_review_eval_gpt4.jsonl --log_file_name code_review_eval_gpt4.log`
7. `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-4 --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_gpt4.jsonl --log_file_name code_smell_eval_gpt4.log`
8. `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-4 --data_load_name code_test_data.jsonl --result_save_name code_test_data_gpt4.jsonl --log_file_name code_test_data_gpt4.log`
