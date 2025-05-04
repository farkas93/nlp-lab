import config

def format_for_sft(record):
    formatter = SquadV2ToSFTFormatter()
    dialog_instance = formatter.preformat_belebele(record=record)
    combined = formatter.combine_prompt_response(dialog_instance=dialog_instance)
    return combined

class SquadV2ToSFTFormatter():
    def preformat_belebele(self, record):
        prompt = ("You are given a context and your task is to answer the question asked only by using the context given." 
                    f"Context:\n {record['context']}\nQuestion: {record['question']}")
        response = record['answers']["text"]
        
        return {
            "prompt": prompt,
            "response": response
        }

    def combine_prompt_response(self, dialog_instance):
        messages = [
            {"role": "user", "content": dialog_instance['prompt']},
            {"role": "assistant", "content": dialog_instance['response']}
        ]
        
        # Use the chat template to format the conversation
        formatted_text = config.apply_chat_template(messages)
        
        return {
            "text": formatted_text
        }
