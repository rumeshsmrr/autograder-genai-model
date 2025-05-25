import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

class CodeEvaluator:
    def __init__(self):
        # Load CodeBERT for evaluation and set it to return hidden states
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert_model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
        self.codebert_model.config.output_hidden_states = True

        # Load CodeT5 for feedback generation
        self.codet5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        self.codet5_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

    def evaluate_criterion_with_codebert(self, ref_code, answer_code, criterion, max_score):
        """
        Evaluate a single criterion using CodeBERT.
        Returns a score (0 to max_score) based on model output.
        """
        prompt = (
            f"Analyze the following Java code and evaluate the criterion: {criterion}\n\n"
            f"Reference Code:\n{ref_code}\n\n"
            f"Student Code:\n{answer_code}\n\n"
            f"Provide a score out of {max_score} based on how well the criterion is met."
        )
        inputs = self.codebert_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.codebert_model(**inputs)
        similarity_score = outputs.logits.softmax(dim=1).tolist()[0][1]  # Extract score from logits

        # Handle identical code explicitly
        if ref_code == answer_code:
            similarity_score = 0.99

        return round(similarity_score * max_score)

    def generate_feedback_with_codet5(self, code, criterion, score):
        """
        Generate feedback for a specific criterion using CodeT5.
        """
        prompt = (
            f"Analyze the following Java code and provide feedback for the criterion: {criterion}\n\n"
            f"Code:\n{code}\n\n"
            f"The score for this criterion is {score}. Explain why this score was assigned.\n\n"
            f"If the criterion is fully met, provide feedback like 'The {criterion.replace('_', ' ')} is implemented correctly.'\n"
            f"If the criterion is partially met, explain what is missing or incorrect.\n"
            f"If the criterion is not met at all, provide feedback like 'The {criterion.replace('_', ' ')} is missing or incorrect.'"
        )
        inputs = self.codet5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.codet5_model.generate(**inputs, max_length=200)
        feedback = self.codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback.strip()

    def compute_code_similarity_with_codebert(self, ref_code, answer_code):
        """
        Compute similarity between two code snippets using CodeBERT embeddings.
        Uses the CLS token from the last hidden state and cosine similarity.
        """
        # Tokenize both code snippets
        inputs_ref = self.codebert_tokenizer(ref_code, return_tensors="pt", truncation=True, padding=True)
        inputs_ans = self.codebert_tokenizer(answer_code, return_tensors="pt", truncation=True, padding=True)

        # Obtain model outputs with hidden states
        outputs_ref = self.codebert_model(**inputs_ref)
        outputs_ans = self.codebert_model(**inputs_ans)

        # Extract CLS embeddings (first token of the last hidden state)
        cls_ref = outputs_ref.hidden_states[-1][:, 0, :]  # Shape: (1, hidden_size)
        cls_ans = outputs_ans.hidden_states[-1][:, 0, :]

        # Compute cosine similarity between the two embeddings
        cos_sim = torch.nn.functional.cosine_similarity(cls_ref, cls_ans).item()
        # Map cosine similarity (range [-1, 1]) to a percentage (0 to 100)
        similarity_percentage = ((cos_sim + 1) / 2) * 100
        similarity_percentage = max(0, min(100, similarity_percentage))
        return round(similarity_percentage, 2)

    def evaluate_submission(self, ref_code, answer_code, rubric, input_data=""):
        """
        Evaluate the student's submission based on a dynamic rubric.
        Returns total score, per-criterion grades and feedback, along with code similarity.
        """
        result = {
            "total_score": 0,
            "grades": {},
            "feedback": {}
        }

        for criterion, max_score in rubric.items():
            # Evaluate criterion score
            score = self.evaluate_criterion_with_codebert(ref_code, answer_code, criterion, max_score)
            # Generate feedback for this criterion
            feedback = self.generate_feedback_with_codet5(answer_code, criterion, score)

            result["grades"][criterion] = score
            result["feedback"][criterion] = feedback
            result["total_score"] += score

        # Compute overall code similarity using CodeBERT
        result["code_similarity_percentage"] = self.compute_code_similarity_with_codebert(ref_code, answer_code)
        result["code_similarity_details"] = {}  # Extend with additional details if needed

        return result
