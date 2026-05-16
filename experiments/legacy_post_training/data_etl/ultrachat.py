import config
import logging

def format_for_dpo(record):
        formatter = UltrachatFormatter()
        combined = formatter.combine_prompt_and_messages(dialog_instance=record)
        return combined

class UltrachatFormatter():

    def combine_prompt_and_messages(self, dialog_instance):
        messages = dialog_instance['messages']
        messages.append({"role": "user", "content": dialog_instance['prompt']})
        
        # Use the chat template to format the conversation
        formatted_text = config.apply_chat_template(messages)

        chosen = dialog_instance['chosen']
        chosen_answer = config.apply_chat_template(chosen)
        logging.info(f"applied template: {chosen_answer}")

        rejected = dialog_instance['rejected']
        rejected_answer = config.apply_chat_template(rejected)
        
        return {
            "prompt": formatted_text,
            "chosen": chosen_answer,
            "rejected": rejected_answer
        }
