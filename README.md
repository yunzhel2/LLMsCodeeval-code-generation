## Tasks: Code Generation (Extension)


Replace "access_token" with your own HuggingFace access token.

Replace "cache_dir" with path to a directory in which a downloaded pretrained model should be cached.

Run following scripts to inference the code generation task.
### Program synthesis
#### GPT3.5 & GPT4
```angular2html
python eval_XXX_cg.py
--api_key
your_apikey
--model
gpt-3.5-turbo-0613
--data_load_name
PATH/program_synthesis_v2.jsonl
--result_save_name
PATH/program_synthesis_eval_XXX.jsonl
--log_file_name
PATH/program_synthesis_eval_XXX.log
```
#### Palm
```angular2html
python eval_palm_cg.py
--api_key
your_apikey
--data_load_name
PATH/program_synthesis_v2.jsonl
--result_save_name
PATH/program_synthesis_eval_palm.jsonl
--log_file_name
PATH/program_synthesis_eval_palm.log
```
#### Others
```angular2html
python eval_XXX_cg.py 
--access_token
access_token
--cache_dir 
cache_dir 
--checkpoint
your_model_ckpt
--data_load_name
PATH/program_synthesis_v2.jsonl
--result_save_name
PATH/program_synthesis_eval_XXX.jsonl
--log_file_name
PATH/program_synthesis_eval_XXX.log
```


### Code Translation
#### GPT3 & GPT4
```angular2html
python eval_XXX_cg.py
--api_key
your_apikey
--model
gpt-3.5-turbo-16k
--data_load_name
PATH/code_translation_v2.jsonl
--result_save_name
PATH/code_translation_eval_XXX.jsonl
--log_file_name
PATH/code_translation_eval_XXX.log
```
#### Palm
```angular2html
python eval_palm_cg.py
--api_key
your_apikey
--data_load_name
PATH/code_translation_v2.jsonl
--result_save_name
PATH/code_translation_eval_palm.jsonl
--log_file_name
PATH/code_translation_eval_palm.log
```
#### Others
```angular2html
python eval_XXX_cg.py 
--access_token
access_token
--cache_dir 
cache_dir 
--checkpoint
your_model_ckpt
--data_load_name
PATH/code_translation_v2.jsonl
--result_save_name
PATH/code_translation_eval_XXX.jsonl
--log_file_name
PATH/code_translation_eval_XXX.log
```



### Code Debugging

#### GPT3.5 & GPT4
```angular2html
python eval_XXX_cg.py
--api_key
your_apikey
--model
gpt-3.5-turbo-16k
--data_load_name
PATH/code_debugging_data.jsonl
--result_save_name
PATH/code_debugging_eval_XXX.jsonl
--log_file_name
PATH/code_debugging_eval_XXX.log
```
#### Palm
```angular2html
python eval_palm_cg.py
--api_key
your_apikey
--data_load_name
PATH/code_debugging_data.jsonl
--result_save_name
PATH/code_debugging_eval_palm.jsonl
--log_file_name
PATH/code_debugging_eval_palm.log
```
#### Others
```angular2html
python eval_XXX_cg.py 
--access_token
access_token
--cache_dir 
cache_dir 
--checkpoint
your_model_ckpt
--data_load_name
PATH/code_debugging_data.jsonl
--result_save_name
PATH/code_debugging_eval_XXX.jsonl
--log_file_name
PATH/code_debugging_eval_XXX.log
```

