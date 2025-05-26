import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import javalang
import json

from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

class CodeEvaluator:
    def __init__(self):
        # Load CodeBERT for embedding extraction (for similarity comparisons)
        self.codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")

        # Load CodeT5 for feedback generation and fallback extraction
        self.codet5_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        self.codet5_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")


    def extract_method_code(self, code, node):
        """
        Extract the complete method code using a simple brace matching algorithm.
        """
        lines = code.splitlines()
        start_line_index = node.position.line - 1
        substring = "\n".join(lines[start_line_index:])
        start_brace_index = substring.find('{')
        if start_brace_index == -1:
            return substring
        brace_count = 0
        end_index = None
        for i, char in enumerate(substring[start_brace_index:], start=start_brace_index):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count == 0:
                end_index = i + 1  # include closing brace
                break
        if end_index is None:
            end_index = len(substring)
        snippet = substring[:end_index]
        return snippet

    def extract_method_snippets(self, code, method_name):
        """
        Try to extract a method (or constructor) by name using javalang.
        Returns None if extraction fails.
        """
        try:
            tree = javalang.parse.parse(code)
            # Look for method declarations
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                if node.name == method_name:
                    return self.extract_method_code(code, node)
            # Look for constructor declarations (if method name is the class name)
            for path, node in tree.filter(javalang.tree.ConstructorDeclaration):
                if node.name == method_name:
                    return self.extract_method_code(code, node)
        except Exception:
            return None
        return None

    def fallback_extract_method_codet5(self, code, method_name):
        """
        Use CodeT5 as a fallback to extract the method snippet from code.
        The prompt instructs CodeT5 to output only the method code.
        """
        prompt = (
            f"Extract the complete code for the Java method named '{method_name}' from the following code. "
            f"Return only the method code including its signature and body.\n\nCode:\n{code}"
        )
        inputs = self.codet5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.codet5_model.generate(**inputs, max_length=300)
        extracted_code = self.codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return extracted_code.strip()

    def compute_code_similarity_codebert(self, ref_code, answer_code):
        """
        Compute cosine similarity between two code snippets using CodeBERT.
        The similarity (in [0, 100]) is based on the cosine similarity of the CLS embeddings.
        """
        inputs_ref = self.codebert_tokenizer(ref_code, return_tensors="pt", truncation=True, padding=True)
        inputs_ans = self.codebert_tokenizer(answer_code, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs_ref = self.codebert_model(**inputs_ref)
            outputs_ans = self.codebert_model(**inputs_ans)
        cls_ref = outputs_ref.last_hidden_state[:, 0, :]
        cls_ans = outputs_ans.last_hidden_state[:, 0, :]
        cos_sim = torch.nn.functional.cosine_similarity(cls_ref, cls_ans).item()
        similarity_percentage = ((cos_sim + 1) / 2) * 100  # map from [-1,1] to [0,100]
        similarity_percentage = max(0, min(100, similarity_percentage))
        return round(similarity_percentage, 2)

    def generate_feedback_with_codet5(self, prompt, max_length=300):
        """
        Use CodeT5 to generate a textual explanation based on the prompt.
        """
        inputs = self.codet5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        outputs = self.codet5_model.generate(**inputs, max_length=max_length)
        feedback = self.codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback.strip()

    def method_exists(self, code, method_name):
        """
        Dynamically check if the code contains an implemented version of a method.
        Uses AST-based analysis if possible, and falls back to regex if needed.
        Checks that the method body is non-empty.
        """
        try:
            tree = javalang.parse.parse(code)
            # Check method declarations
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                if node.name == method_name:
                    if node.body and len(node.body) > 0:
                        return True
            # Check constructor declarations
            for path, node in tree.filter(javalang.tree.ConstructorDeclaration):
                if node.name == method_name:
                    if node.body and len(node.body) > 0:
                        return True
            return False
        except Exception:
            # Fallback: use regex to look for a plausible signature
            pattern = rf'\b(public|private|protected)\s+[\w<>\[\]]+\s+{re.escape(method_name)}\s*\('
            constructor_pattern = rf'\b(public|private|protected)\s+{re.escape(method_name)}\s*\('
            return bool(re.search(pattern, code) or re.search(constructor_pattern, code))

    def evaluate_method_pair(self, method_name, ref_code, answer_code, max_score):
        """
        Evaluate a method-specific criterion for answer_code.
        First, dynamically check if answer_code implements the method.
        If not, return 0 points. Otherwise, extract the method implementations
        (using javalang or CodeT5 fallback) and compare the answer snippet to the reference.
        Generates an explanation via CodeT5.
        Returns a tuple (score, explanation).
        """
        # Check if answer_code has an implemented version of the method
        if not self.method_exists(answer_code, method_name):
            return 0, f"Method '{method_name}' is missing or has no implementation in the answer submission."

        # Extract the reference and answer snippets
        ref_snippet = self.extract_method_snippets(ref_code, method_name)
        if not ref_snippet or len(ref_snippet.strip()) < 20:
            ref_snippet = self.fallback_extract_method_codet5(ref_code, method_name)
        answer_snippet = self.extract_method_snippets(answer_code, method_name)
        if not answer_snippet or len(answer_snippet.strip()) < 20:
            answer_snippet = self.fallback_extract_method_codet5(answer_code, method_name)

        # If after fallback we still get too short a snippet, treat it as missing
        if not ref_snippet or len(ref_snippet.strip()) < 20 or not answer_snippet or len(answer_snippet.strip()) < 20:
            return 0, f"Method '{method_name}' could not be properly extracted from one of the submissions."

        similarity = self.compute_code_similarity_codebert(ref_snippet, answer_snippet)
        score = round((similarity / 100) * max_score)

        prompt = (
            f"Compare the following two implementations of the Java method '{method_name}' and explain the differences.\n\n"
            f"Reference implementation:\n{ref_snippet}\n\n"
            f"Student's implementation:\n{answer_snippet}\n\n"
            f"Explain the differences and potential issues in the student's implementation."
        )
        explanation = self.generate_feedback_with_codet5(prompt, max_length=300)
        return score, explanation

    def evaluate_overall_criterion(self, ref_code, answer_code, criterion, max_score):
        """
        For overall criteria (like output match or code quality), compare the full reference and answer code.
        Uses CodeBERT for similarity and CodeT5 for explanation.
        """
        prompt = (
            f"Compare the following Java code for the criterion '{criterion}'.\n\n"
            f"Reference Code:\n{ref_code}\n\n"
            f"Student Code:\n{answer_code}\n\n"
            f"Explain the differences and issues with respect to the criterion '{criterion}'."
        )
        explanation = self.generate_feedback_with_codet5(prompt, max_length=300)
        similarity = self.compute_code_similarity_codebert(ref_code, answer_code)
        score = round((similarity / 100) * max_score)
        return score, explanation

    def validate_syntax(self, code):
        """
        Validate Java syntax using javalang.
        """
        try:
            javalang.parse.parse(code)
            return True
        except Exception:
            return False

    def evaluate_submission(self, ref_code, answer_code, rubric, input_data=""):
        """
        Evaluate the answer submission based on the provided rubric.
        For method-specific criteria, if the answer implements the method (checked dynamically via AST),
        its implementation is extracted and compared to the reference.
        For overall criteria, the entire answer code is compared with the reference.
        Returns a detailed result including scores and explanations.
        """
        result = {
            "total_score": 0,
            "grades": {},
            "feedback": {},
            "code_similarity_percentage": self.compute_code_similarity_codebert(ref_code, answer_code),
            "code_similarity_details": {},
            "syntax_errors": []
        }

        if not self.validate_syntax(answer_code):
            result["syntax_errors"].append("The submitted code contains syntax errors.")

        for criterion, max_score in rubric.items():
            score = 0
            feedback = ""

            if criterion in ["default_constructor", "overloaded_constructor"]:
                score, feedback = self.evaluate_method_pair("Employee", ref_code, answer_code, max_score)
            elif criterion == "read_method":
                score, feedback = self.evaluate_method_pair("read", ref_code, answer_code, max_score)
            elif criterion == "display_method":
                score, feedback = self.evaluate_method_pair("display", ref_code, answer_code, max_score)
            elif criterion == "calculate_annual_salary_method":
                score, feedback = self.evaluate_method_pair("calculateAnnualSalary", ref_code, answer_code, max_score)
            elif criterion == "calculate_tax_method":
                score, feedback = self.evaluate_method_pair("calculateTax", ref_code, answer_code, max_score)
            elif criterion in ["output_match", "code_quality"]:
                score, feedback = self.evaluate_overall_criterion(ref_code, answer_code, criterion, max_score)
            else:
                score, feedback = self.evaluate_overall_criterion(ref_code, answer_code, criterion, max_score)

            result["grades"][criterion] = score
            result["feedback"][criterion] = feedback
            result["total_score"] += score

        return result
