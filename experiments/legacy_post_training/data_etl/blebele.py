import config

def format_for_sft(record):
    formatter = BelebeleToSFTFormatter()
    dialog_instance = formatter.preformat_belebele(record=record)
    combined = formatter.combine_prompt_response(dialog_instance=dialog_instance)
    return combined

class BelebeleToSFTFormatter():
    def preformat_belebele(self, record):
        prompt = ("Your task is to identify the correct answer within the passage. " 
                    f"The language is {self.convert_lang_tag(record['dialect'])}. "
                    f"Passage: {record['flores_passage']}\n\nQuestion: {record['question']}\n\nChoices:\n")
        prompt += f"1. {record['mc_answer1']}\n"
        prompt += f"2. {record['mc_answer2']}\n"
        prompt += f"3. {record['mc_answer3']}\n"
        prompt += f"4. {record['mc_answer4']}\n"
        
        prompt += "\nProvide the correct answer label (1, 2, 3, or 4)."
        
        response = f"The correct answer is {record['correct_answer_num']}."
        
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
    
    def convert_lang_tag(self, lang_tag):
        if lang_tag == "acm_Arab":
          return "arabic"
        if lang_tag == "ita_Latn":
            return "italian"
        if lang_tag == "fra_Latn":
            return "french"
        if lang_tag == "hun_Latn":
            return "hungarian"
        if lang_tag == "deu_Latn":
            return "german"
        if lang_tag == "rus_Cyrl":
            return "russian"
        if lang_tag == "spa_Latn":
            return "spanish"
        if lang_tag == "eng_Latn":
            return "english"
        # If none applies
        return "randomly selected"
