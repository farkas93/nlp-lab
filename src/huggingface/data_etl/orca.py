import config
import logging

def format_for_sft(record):
    formatter = OrcaFormatter()
    combined = formatter.slim_orca_combine_prompt_response_sft(dialog_instance=record)
    return combined

def format_for_dpo(record):
    formatter = OrcaFormatter()
    combined = formatter.orca_dpo_combine(record=record)
    return combined

class OrcaFormatter():
    
    def orca_dpo_combine(self, record):
        messages = [{"role": "system", "content": record['system']},
                    {"role": "user", "content": record['question']},
                    ]
        formatted_prompt=self.apply_chat_template(messages)
        return {
            "prompt": formatted_prompt,
            "chosen": record['chosen'],
            "rejected": record['rejected']
        }


    def slim_orca_combine_prompt_response_sft(self, dialog_instance):
        messages = []
        for m in dialog_instance['conversations']:
            messages.append({"role": self.role_translation(m['from']), "content": m['value']})
        
        formatted_text=self.apply_chat_template(messages)
        return {
            "text": formatted_text
        }
    

    def apply_chat_template(self, messages):
        # Use the chat template to format the conversation
        try:
            formatted_dialog = config.apply_chat_template(messages)
        except Exception as e:
            logging.debug(e)
            logging.debug("Merging system and user prompt instead")
            sys = messages[0]
            messages.pop(0)
            messages[0] = self.merge_sys_and_user(sys, messages[0])
            formatted_dialog = config.apply_chat_template(messages)
        return formatted_dialog
    
    def merge_sys_and_user(self, sys_msg, usr_msg):
        if sys_msg['role'] == 'system' and usr_msg['role'] == 'user':
            merge_msg = {"role": "user", "content": f"System: {sys_msg['content']}\n{usr_msg['content']}"}
            return merge_msg


    def role_translation(self, role):
        if role == "system":
            return role
        if role == "human":
            return "user"
        if role == "gpt":
            return "assistant"
        return "unkown role"
