import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import subprocess
import javalang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fixed criteria with dedicated evaluation methods
FIXED_CRITERIA = {
    "syntax_correctness", 
    "output_match",
    "code_similarity",
    "error_handling",
    "boundary_conditions"
}

# Load models
codebert_name = "microsoft/codebert-base"
codet5_name = "Salesforce/codet5-small"

codebert_tokenizer = AutoTokenizer.from_pretrained(codebert_name)
codebert_model = AutoModelForSequenceClassification.from_pretrained(codebert_name)
codet5_tokenizer = AutoTokenizer.from_pretrained(codet5_name)
codet5_model = AutoModelForSeq2SeqLM.from_pretrained(codet5_name)

def normalize_code(code):
    """Ultra-strict normalization for exact match comparison"""
    return "\n".join(line.strip() for line in code.splitlines() if line.strip()).strip()

def evaluate_code_similarity(ref_code, ans_code):
    """Hybrid similarity evaluation with exact match check"""
    ref_norm = normalize_code(ref_code)
    ans_norm = normalize_code(ans_code)
    
    # Exact match shortcut
    if ref_norm == ans_norm:
        return 100.0, 100.0, 100.0

    # TF-IDF similarity
    vectorizer = TfidfVectorizer().fit_transform([ref_norm, ans_norm])
    vectors = vectorizer.toarray()
    tfidf_sim = cosine_similarity(vectors)[0, 1] * 100

    # CodeBERT semantic similarity
    inputs = codebert_tokenizer(
        [ref_code, ans_code],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = codebert_model(**inputs)
    codebert_sim = torch.softmax(outputs.logits, dim=1)[0][1].item() * 100

    # Combined similarity
    final_sim = (tfidf_sim + codebert_sim) / 2
    return round(tfidf_sim, 2), round(codebert_sim, 2), round(final_sim, 2)

def detect_syntax_errors_with_javalang(code_snippet):
    """
    Validates Java syntax using javalang.
    Returns a list of syntax errors.
    """
    errors = []
    try:
        javalang.parse.parse(code_snippet)
    except javalang.parser.JavaSyntaxError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
    return errors

def detect_syntax_errors_with_compiler(code_snippet, filename="Temp"):
    """
    Compiles Java code to identify syntax errors, ignoring file name issues.
    """
    filename = f"{filename}.java"
    with open(filename, "w") as f:
        f.write(code_snippet)

    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    if compile_process.returncode != 0:
        errors = compile_process.stderr.splitlines()
        # Filter out file name issues
        filtered_errors = [
            error for error in errors if "should be declared in a file named" not in error
        ]
        return filtered_errors  # Return remaining errors
    return []  # No errors

def evaluate_syntax(code_snippet, rubric_weight):
    """
    Combines prompt-based and rule-based syntax evaluations.
    """
    max_score = rubric_weight
    error_penalty = 0.5  # Deduction per error
    errors = []

    # Rule-based syntax validation
    compiler_errors = detect_syntax_errors_with_compiler(code_snippet)
    if compiler_errors:
        errors.extend(compiler_errors)
        max_score -= len(compiler_errors) * error_penalty

    javalang_errors = detect_syntax_errors_with_javalang(code_snippet)
    if javalang_errors:
        errors.extend(javalang_errors)
        max_score -= len(javalang_errors) * error_penalty

    return max(0, max_score), errors

def run_java_code(code, input_data="", filename="Temp"):
    """
    Compiles and runs Java code, handling input data.
    """
    filename = f"{filename}.java"
    try:
        with open(filename, "w") as f:
            f.write(code)

        # Compile Java code
        compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
        if compile_process.returncode != 0:
            return f"Compilation Error: {compile_process.stderr.strip()}"

        # Run Java program
        run_process = subprocess.run(
            ["java", filename.replace(".java", "")],
            input=input_data,
            text=True,
            capture_output=True
        )
        if run_process.returncode != 0:
            return f"Runtime Error: {run_process.stderr.strip()}"

        return run_process.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    
def compare_outputs(ref_output, ans_output):
    """
    Compares outputs line by line and calculates a match percentage.
    """
    ref_lines = [line.strip() for line in ref_output.splitlines() if line.strip()]
    ans_lines = [line.strip() for line in ans_output.splitlines() if line.strip()]

    total_lines = max(len(ref_lines), len(ans_lines))
    matching_lines = sum(1 for ref, ans in zip(ref_lines, ans_lines) if ref == ans)

    return (matching_lines / total_lines) * 100 if total_lines > 0 else 0.0

def parse_constructors_and_methods(code):
    """Static analysis for method/constructor detection. Returns constructors and methods."""
    try:
        tree = javalang.parse.parse(code)
        constructors = []
        methods = []
        
        for path, node in tree:
            if isinstance(node, javalang.tree.ConstructorDeclaration):
                params = [f"{p.type.name} {p.name}" for p in node.parameters]
                constructors.append({
                    "name": node.name,
                    "params": params,
                    "is_default": len(params) == 0,
                    "node": node
                })
            elif isinstance(node, javalang.tree.MethodDeclaration):
                params = [f"{p.type.name} {p.name}" for p in node.parameters]
                methods.append({
                    "name": node.name,
                    "params": params,
                    "return_type": node.return_type.name if node.return_type else "void",
                    "node": node
                })
        return constructors, methods
    except Exception as e:
        return [], []

def extract_method_body(code, method_name):
    """
    Attempts to extract the method body of the given method name from the Java code.
    This is a simplified extraction using the node's string representation.
    """
    try:
        _, methods = parse_constructors_and_methods(code)
        for method in methods:
            if method["name"].lower() == method_name.lower():
                # Return the string representation of the method node as a proxy for its implementation.
                return str(method["node"])
    except Exception as e:
        return ""
    return ""

def evaluate_method_correctness(reference_code, answer_code, method_name, threshold=80):
    """
    Evaluates whether the method implementation in answer_code is correct by comparing it
    to the reference_code. Returns 100 if the similarity is above the threshold, otherwise returns the similarity percentage.
    """
    ref_body = extract_method_body(reference_code, method_name)
    ans_body = extract_method_body(answer_code, method_name)
    if not ref_body or not ans_body:
        return 0  # If either method is missing, return 0
    # Use the existing similarity function on the method bodies
    _, _, similarity = evaluate_code_similarity(ref_body, ans_body)
    return 100 if similarity >= threshold else similarity

def evaluate_structural_criteria(reference_code, answer_code, criteria):
    """Evaluate method/constructor correctness using static analysis and similarity checks."""
    results = {}
    
    for criterion in criteria:
        criterion_lower = criterion.lower()
        
        if "constructor" in criterion_lower:
            # For constructors, check for presence using parse_constructors_and_methods
            constructors, _ = parse_constructors_and_methods(answer_code)
            if "default" in criterion_lower:
                results[criterion] = 100 if any(ctor["is_default"] for ctor in constructors) else 0
            elif "overloaded" in criterion_lower:
                results[criterion] = 100 if len(constructors) > 1 else 0
        elif "method" in criterion_lower:
            # For methods, evaluate correctness by comparing the method bodies.
            # Extract the method name from criterion (e.g., "Read() method" -> "read")
            method_name = criterion.split('(')[0].strip()
            results[criterion] = evaluate_method_correctness(reference_code, answer_code, method_name)
    return results

def evaluate_with_codet5(code, criteria):
    """LLM-based evaluation for abstract criteria"""
    results = {}
    for criterion in criteria:
        prompt = f"""Does this Java code satisfy: {criterion}?
        Code:
        \"\"\"{code}\"\"\"
        Answer Yes or No:"""
        
        try:
            inputs = codet5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = codet5_model.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=1,
                early_stopping=True
            )
            response = codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[criterion] = 100 if "yes" in response.lower() else 0
        except Exception as e:
            results[criterion] = 0
    return results

def process_rubric_weights(rubric):
    """Extract weights from rubric structure"""
    weights = {}
    for key, value in rubric.items():
        if isinstance(value, dict):
            weights[key] = value.get("weight", 0)
        else:
            weights[key] = value
    return weights

def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    # Process rubric weights
    rubric_weights = process_rubric_weights(rubric)
    
    # Phase 1: Code similarity check (overall)
    _, _, code_sim = evaluate_code_similarity(reference_code, answer_code)
    if code_sim == 100:
        valid_criteria = {k: v for k, v in rubric_weights.items() if v > 0}
        total_score = sum(valid_criteria.values())
        final_score = min(total_score, 100)
        return {
            "final_score": round(final_score, 2),
            "grades": {k: v for k, v in valid_criteria.items()},
            "code_similarity_percentage": 100.0,
            "syntax_errors": []
        }

    # Phase 2: Syntax validation
    syntax_score, syntax_errors = evaluate_syntax(answer_code, rubric_weights.get("syntax_correctness", 0))
    
    # Phase 3: Output correctness
    ref_output = run_java_code(reference_code, input_data)
    ans_output = run_java_code(answer_code, input_data)
    output_match = compare_outputs(ref_output, ans_output) * (rubric_weights.get("output_match", 0) / 100)

    # Phase 4: Dynamic criteria evaluation
    # Evaluate criteria that are not fixed; now include method correctness comparison
    dynamic_criteria = [k for k in rubric_weights if k not in FIXED_CRITERIA]
    structural_results = evaluate_structural_criteria(reference_code, answer_code, dynamic_criteria)
    remaining_criteria = [c for c in dynamic_criteria if c not in structural_results]
    llm_results = evaluate_with_codet5(answer_code, remaining_criteria)
    
    # Combine results
    grades = {
        "syntax_correctness": syntax_score,
        "output_match": output_match,
        "code_similarity": code_sim * (rubric_weights.get("code_similarity", 0) / 100),
        **{k: v * (rubric_weights[k] / 100) for k, v in {**structural_results, **llm_results}.items()}
    }
    
    # Calculate final score
    valid_grades = {k: v for k, v in grades.items() if rubric_weights.get(k, 0) > 0}
    final_score = min(100, sum(valid_grades.values()))
    
    return {
        "final_score": round(final_score, 2),
        "grades": {k: round(v, 2) for k, v in valid_grades.items()},
        "code_similarity_percentage": round(code_sim, 2),
        "syntax_errors": syntax_errors
    }
