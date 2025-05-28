import re

class GSM8KRewards():
    def __init__(self, rstart, rstop, solstart, solstop):
        self.reasoning_start = rstart
        self.reasoning_end = rstop
        self.solution_start = solstart
        self.solution_end = solstop
        self.match_numbers = re.compile(
            rf"{self.solution_start}.*?([\d\.]{{1,}})",
            flags = re.MULTILINE | re.DOTALL
        )
        self.match_format = re.compile(
            rf"^[\s]{{0,}}"\
            rf"{self.reasoning_start}.+?{self.reasoning_end}.*?"\
            rf"{self.solution_start}(.+?){self.solution_end}"\
            rf"[\s]{{0,}}$",
            flags = re.MULTILINE | re.DOTALL
        )

    def match_format_exactly(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if self.match_format.search(response) is not None: score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!
            score += 0.5 if response.count(self.reasoning_start) == 1 else -0.5
            score += 0.5 if response.count(self.reasoning_end)   == 1 else -0.5
            score += 0.5 if response.count(self.solution_start)  == 1 else -0.5
            score += 0.5 if response.count(self.solution_end)    == 1 else -0.5
            scores.append(score)
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            # Correct answer gets 3 points!
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                    elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                    else: score -= 1.0 # Penalize wrong answers
                except:
                    score -= 0.5 # Penalize
            scores.append(score)
        return scores
    

    def check_numbers(self, prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                guess       = float(guess.strip())
                scores.append(1.5 if guess == true_answer else 0.0)
            except:
                scores.append(0)
                continue
        return scores
    
    def get(self):
        return [
            self.match_format_exactly,
            self.match_format_approximately,
            self.check_answer,
            self.check_numbers,
        ]