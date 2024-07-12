import config 
import logging

def format_for_sft(record):
    formatter = Ai2arcToSFTFormatter()
    dialog_instance = formatter.preformat_ai2_arc(record=record)
    combined = formatter.combine_prompt_response(dialog_instance=dialog_instance) 
    logging.info(combined)
    return combined



class Ai2arcToSFTFormatter():
        
    def preformat_ai2_arc(self, record):
        prompt = f"Question: {record['question']}\n\nChoices:\n"
        
        choices = record['choices']
        for label, text in zip(choices['label'], choices['text']):
            prompt += f"{label}. {text}\n"
        
        prompt += "\nProvide the correct answer label (A, B, C, or D)."
        
        response = f"The correct answer is {record['answerKey']}."
        
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
