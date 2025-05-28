from rewards.gsm8k import GSM8KRewards
from datasets import load_dataset


def format_for_grpo(record, rewards):
    formatter = GSM8KFormatter(rewards)
    return formatter.to_grpo(record)


class GSM8KFormatter():

    def __init__(self, sysp: str, rewards: GSM8KRewards):
        self.rewards = rewards
        self.system_prompt = sysp
        
           

    def extract_hash_answer(self, text):
        if "####" not in text: return None
        return text.split("####")[1].strip()
    
    def test_format(self):
        test_re = self.rewards.match_format.search(
            f"{self.rewards.reasoning_start}Let me think!{self.rewards.reasoning_end}"\
            f"{self.rewards.solution_start}2{self.rewards.solution_end}",
        )
        print(test_re)

    def to_grpo(self, record):
        return {
            "prompt" : [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": record["question"]},
            ],
            "answer": self.extract_hash_answer(record["answer"]),
        }
        

    def get_grpo_train(self):
        dataset = load_dataset("openai/gsm8k", "main", split = "train")
        return dataset.map(lambda x: self.to_grpo(x))

    def get_grpo_test(self):
        dataset = load_dataset("openai/gsm8k", "main", split = "test")
        return dataset.map(lambda x: self.to_grpo(x))