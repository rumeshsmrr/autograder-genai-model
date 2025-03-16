from evaluate import evaluate_model_accuracy_and_f1

if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "dataset/large_test_dataset.csv"

    # Define the rubric
    rubric = {
        "syntax_correctness": 30,
        "output_match": 40,
        "code_similarity": 30,
    }

    # Run the evaluation
    accuracy, f1 = evaluate_model_accuracy_and_f1(dataset_path, rubric)

    # Display the results
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model F1-Score: {f1:.2f}")
