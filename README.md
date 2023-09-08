## Tasks: code review & code smell & code test

### LLM Evaluation

* [X]  GPT-3.5
* [X]  GPT-4
* [X]  PaLM
* [ ]  CodeLLaMA
* [ ]  Vicuna
* [ ]  LLaMA2
* [ ]  LLaMA
* [ ]  WizardCoder
* [ ]  StarCoder
* [ ]  CodeT5+

### Installation

1. `cd code-data-eval`
2. `pip install -r requirements.txt`

### GPT-3.5 ✔

Replace "openai_api_key" with your own OpenAI API key.

1. For code review: `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_review_data.jsonl --result_save_name code_review_eval_gpt3.jsonl --log_file_name code_review_eval_gpt3.log`
2. For code smell: `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_gpt3.jsonl --log_file_name code_smell_eval_gpt3.log`
3. For code test: `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-3.5-turbo-0613 --data_load_name code_test_data.jsonl --result_save_name code_test_data_gpt3.jsonl --log_file_name code_test_data_gpt3.log`

### GPT-4 ✔

Replace "openai_api_key" with your own OpenAI API key.

1. For code review: `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name code_review_data.jsonl --result_save_name code_review_eval_gpt4.jsonl --log_file_name code_review_eval_gpt4.log`
2. For code smell: `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_gpt4.jsonl --log_file_name code_smell_eval_gpt4.log`
3. For code test: `python scripts/eval_gpt.py --api_key openai_api_key --model gpt-4-0613 --data_load_name code_test_data.jsonl --result_save_name code_test_data_gpt4.jsonl --log_file_name code_test_data_gpt4.log`

### PaLM ✔

Replace "google_api_key" with your own Google API key.

1. For code review: `python scripts/eval_palm.py --api_key google_api_key --data_load_name code_review_data.jsonl --result_save_name code_review_eval_palm.jsonl --log_file_name code_review_eval_palm.log`
2. For code smell: `python scripts/eval_palm.py --api_key google_api_key --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_palm.jsonl --log_file_name code_smell_eval_palm.log`
3. For code test: `python scripts/eval_palm.py --api_key google_api_key --data_load_name code_test_data.jsonl --result_save_name code_test_data_palm.jsonl --log_file_name code_test_data_palm.log`

### CodeLLaMA ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_codellama.py --access_token access_token --checkpoint codellama/CodeLlama-34b-Instruct-hf --data_load_name code_review_data.jsonl --result_save_name code_review_eval_codellama.jsonl --log_file_name code_review_eval_codellama.log`
2. For code smell: `python scripts/eval_codellama.py --access_token access_token --checkpoint codellama/CodeLlama-34b-Instruct-hf --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_codellama.jsonl --log_file_name code_smell_eval_codellama.log`
3. For code test: `python scripts/eval_codellama.py --access_token access_token --checkpoint codellama/CodeLlama-34b-Instruct-hf --data_load_name code_test_data.jsonl --result_save_name code_test_data_codellama.jsonl --log_file_name code_test_data_codellama.log`

### Vicuna ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_vicuna.py --access_token access_token --checkpoint lmsys/vicuna-13b-v1.5-16k --data_load_name code_review_data.jsonl --result_save_name code_review_eval_vicuna.jsonl --log_file_name code_review_eval_vicuna.log`
2. For code smell: `python scripts/eval_vicuna.py --access_token access_token --checkpoint lmsys/vicuna-13b-v1.5-16k --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_vicuna.jsonl --log_file_name code_smell_eval_vicuna.log`
3. For code test: `python scripts/eval_vicuna.py --access_token access_token --checkpoint lmsys/vicuna-13b-v1.5-16k --data_load_name code_test_data.jsonl --result_save_name code_test_data_vicuna.jsonl --log_file_name code_test_data_vicuna.log`

### LLaMA2 ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_llama2.py --access_token access_token --checkpoint meta-llama/Llama-2-70b-chat-hf --data_load_name code_review_data.jsonl --result_save_name code_review_eval_llama2.jsonl --log_file_name code_review_eval_llama2.log`
2. For code smell: `python scripts/eval_llama2.py --access_token access_token --checkpoint meta-llama/Llama-2-70b-chat-hf --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_llama2.jsonl --log_file_name code_smell_eval_llama2.log`
3. For code test: `python scripts/eval_llama2.py --access_token access_token --checkpoint meta-llama/Llama-2-70b-chat-hf --data_load_name code_test_data.jsonl --result_save_name code_test_data_llama2.jsonl --log_file_name code_test_data_llama2.log`

### LLaMA ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_llama.py --access_token access_token --checkpoint huggyllama/llama-65b --data_load_name code_review_data.jsonl --result_save_name code_review_eval_llama.jsonl --log_file_name code_review_eval_llama.log`
2. For code smell: `python scripts/eval_llama.py --access_token access_token --checkpoint huggyllama/llama-65b --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_llama.jsonl --log_file_name code_smell_eval_llama.log`
3. For code test: `python scripts/eval_llama.py --access_token access_token --checkpoint huggyllama/llama-65b --data_load_name code_test_data.jsonl --result_save_name code_test_data_llama.jsonl --log_file_name code_test_data_llama.log`

### WizardCoder ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_wizardcoder.py --access_token access_token --checkpoint WizardLM/WizardCoder-15B-V1.0 --data_load_name code_review_data.jsonl --result_save_name code_review_eval_wizardcoder.jsonl --log_file_name code_review_eval_wizardcoder.log`
2. For code smell: `python scripts/eval_wizardcoder.py --access_token access_token --checkpoint WizardLM/WizardCoder-15B-V1.0 --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_wizardcoder.jsonl --log_file_name code_smell_eval_wizardcoder.log`
3. For code test: `python scripts/eval_wizardcoder.py --access_token access_token --checkpoint WizardLM/WizardCoder-15B-V1.0--data_load_name code_test_data.jsonl --result_save_name code_test_data_wizardcoder.jsonl --log_file_name code_test_data_wizardcoder.log`

### StarCoder ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_starcoder.py --access_token access_token --checkpoint bigcode/starcoderplus --data_load_name code_review_data.jsonl --result_save_name code_review_eval_starcoder.jsonl --log_file_name code_review_eval_starcoder.log`
2. For code smell: `python scripts/eval_starcoder.py --access_token access_token --checkpoint bigcode/starcoderplus --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_starcoder.jsonl --log_file_name code_smell_eval_starcoder.log`
3. For code test: `python scripts/eval_starcoder.py --access_token access_token --checkpoint bigcode/starcoderplus --data_load_name code_test_data.jsonl --result_save_name code_test_data_starcoder.jsonl --log_file_name code_test_data_starcoder.log`

### CodeT5+ ❌

Replace "access_token" with your own HuggingFace access token.

1. For code review: `python scripts/eval_codet5p.py --access_token access_token --checkpoint Salesforce/instructcodet5p-16b --data_load_name code_review_data.jsonl --result_save_name code_review_eval_codet5p.jsonl --log_file_name code_review_eval_codet5p.log`
2. For code smell: `python scripts/eval_codet5p.py --access_token access_token --checkpoint Salesforce/instructcodet5p-16b --data_load_name code_smell_data.jsonl --result_save_name code_smell_eval_codet5p.jsonl --log_file_name code_smell_eval_codet5p.log`
3. For code test: `python scripts/eval_codet5p.py --access_token access_token --checkpoint Salesforce/instructcodet5p-16b --data_load_name code_test_data.jsonl --result_save_name code_test_data_codet5p.jsonl --log_file_name code_test_data_codet5p.log`
