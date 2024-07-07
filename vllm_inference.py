import time
from vllm import LLM, SamplingParams
import config
from transformers import AutoTokenizer

def download_and_test(model, max_tokens):

    prompts = [
        "Write an infinite story about a boy and girl and their adventures!"
    ]
    # Before your existing code
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Before generating
    for prompt in prompts:
        input_tokens = tokenizer.encode(prompt)
        print(f"Input tokens: {len(input_tokens)}")

    sampling_params = SamplingParams(temperature=0.1, top_p=0.8, top_k=20, max_tokens=max_tokens)
    loading_start = time.time()
    llm = LLM(model=model, max_model_len=8192)
    print("--- Loading time: %s seconds ---" % (time.time() - loading_start))
    generation_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print("--- Generation time: %s seconds ---" % (time.time() - generation_time))
    # After generating
    for output in outputs:
        generated_text = output.outputs[0].text
        output_tokens = tokenizer.encode(generated_text)
        print(f"Output tokens: {len(output_tokens)}")
        print(generated_text)
        print('------')
    
if __name__ == "__main__":
    max_tokens=4096
    download_and_test(model=config.BASE_MODEL, max_tokens=max_tokens)