import config

def format_for_dpo(record):
    ds_conf = config.DPO_DATASETS['allenai/ultrafeedback_binarized_cleaned']
    if ds_conf['combine_msgs']:
        formatter = UltrachatFormatter()
        combined = formatter.combine_prompt_response(dialog_instance=record)
        return combined
    else:
        return record

class UltrachatFormatter():

    def combine_prompt_and_messages(self, dialog_instance):
        messages = dialog_instance['messages']
        messages.append({"role": "user", "content": dialog_instance['prompt']})
        
        # Use the chat template to format the conversation
        formatted_text = config.apply_chat_template(messages)
        
        return {
            "prompt": formatted_text,
            "chosen": dialog_instance['chosen'],
            "rejected": dialog_instance['rejected']
        }
