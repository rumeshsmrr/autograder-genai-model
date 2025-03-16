import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import subprocess
import javalang

# Load CodeBERT model for code similarity
codebert_name = "microsoft/codebert-base"
codebert_tokenizer = AutoTokenizer.from_pretrained(codebert_name)
codebert_model = AutoModelForSequenceClassification.from_pretrained(codebert_name)

# Normalize code for better comparison
def normalize_code(code):
    return "\n".join(line.strip() for line in code.splitlines() if line.strip())

# Syntax validation using javalang
def detect_syntax_errors_with_javalang(code_snippet):
    errors = []
    try:
        javalang.parse.parse(code_snippet)
    except javalang.parser.JavaSyntaxError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
    return errors

# Syntax validation using javac compiler
def detect_syntax_errors_with_compiler(code_snippet, filename="Temp"):
    filename = f"{filename}.java"
    with open(filename, "w") as f:
        f.write(code_snippet)

    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    if compile_process.returncode != 0:
        errors = compile_process.stderr.splitlines()
        return errors
    return []

# Evaluate syntax correctness
def evaluate_syntax(code_snippet, rubric_weight):
    max_score = rubric_weight
    error_penalty = 0.5
    errors = []

    compiler_errors = detect_syntax_errors_with_compiler(code_snippet)
    if compiler_errors:
        errors.extend(compiler_errors)
        max_score -= len(compiler_errors) * error_penalty

    javalang_errors = detect_syntax_errors_with_javalang(code_snippet)
    if javalang_errors:
        errors.extend(javalang_errors)
        max_score -= len(javalang_errors) * error_penalty

    return max(0, max_score), errors

# Evaluate code similarity
def evaluate_code_similarity(ref_code, ans_code):
    ref_code_normalized = normalize_code(ref_code)
    ans_code_normalized = normalize_code(ans_code)

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer().fit_transform([ref_code_normalized, ans_code_normalized])
    vectors = vectorizer.toarray()
    tfidf_similarity = cosine_similarity(vectors)[0, 1] * 100

    # CodeBERT Similarity
    inputs = codebert_tokenizer(
        f"Determine similarity:\nCode 1:\n{ref_code}\nCode 2:\n{ans_code}",
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = codebert_model(**inputs)
    codebert_similarity = torch.softmax(outputs.logits, dim=1)[0][1].item() * 100

    final_similarity = (tfidf_similarity + codebert_similarity) / 2
    return round(tfidf_similarity, 2), round(codebert_similarity, 2), round(final_similarity, 2)

# Compare program outputs
def compare_outputs(ref_output, ans_output):
    ref_lines = [line.strip() for line in ref_output.splitlines() if line.strip()]
    ans_lines = [line.strip() for line in ans_output.splitlines() if line.strip()]

    total_lines = max(len(ref_lines), len(ans_lines))
    matching_lines = sum(1 for ref, ans in zip(ref_lines, ans_lines) if ref == ans)

    return (matching_lines / total_lines) * 100 if total_lines > 0 else 0.0

# Evaluate the entire model
def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    syntax_score, syntax_errors = evaluate_syntax(answer_code, rubric.get("syntax_correctness", 10))
    tfidf_similarity, codebert_similarity, final_similarity = evaluate_code_similarity(reference_code, answer_code)
    output_match = 100  # Simplified: assume perfect match for outputs

    results = {
        "syntax_correctness": syntax_score,
        "output_match_percentage": output_match * rubric.get("output_match", 40) / 100,
        "code_similarity": final_similarity * rubric.get("code_similarity", 30) / 100,
    }

    final_score = sum(results.values())
    return {
        "final_score": round(final_score, 2),
        "results": results,
        "syntax_errors": syntax_errors,
        "code_similarity_details": {
            "TF-IDF Similarity": tfidf_similarity,
            "CodeBERT Similarity": codebert_similarity,
            "Final Aggregated Similarity": final_similarity,
        },
    }

# Evaluate model's accuracy and F1-score
def evaluate_model_accuracy_and_f1(dataset_path, rubric, similarity_threshold=70):
    dataset = pd.read_csv(dataset_path)
    ground_truth = []
    predictions = []

    for _, row in dataset.iterrows():
        reference_code = row["reference_code"]
        answer_code = row["answer_code"]
        ground_truth_score = row["ground_truth_score"]

        evaluation = evaluate_code(reference_code, answer_code, rubric=rubric)
        predicted_score = 1 if evaluation["final_score"] >= similarity_threshold else 0

        ground_truth.append(ground_truth_score)
        predictions.append(predicted_score)

    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average="binary")
    return accuracy, f1
