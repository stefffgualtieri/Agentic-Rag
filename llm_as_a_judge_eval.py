#$6.10

from openai import OpenAI
import json

client = OpenAI()



with open("NaiveRAG_2_raw_model.json", "r") as f:
	rows = json.load(f)
	result = []
	for row, i in zip(rows, range(len(rows))):
		print(f"Processing question {i+1}/{len(rows)}")
		question = row["question"]
		correct_answer = row["true_answer"]
		model_answer = row["model_answer"]

		instruction = f"""

		You will be given a question, the correct answer and the answer provided by a language model. Your task is to evaluate
		the answer given by the model and give it a score from 1 to 4. 
		Here is the scale you should use to build your answer:
		1: The Model answer is terrible: completely incorrect
		2: The Model answer is mostly not helpful: misses some key aspects of the question or misses some parts of the answer
		3: The Model answer is mostly helpful: it is correct but still less informative than the correct answer
		4: The Model answer is excellent: it provides the exact same information as the correct answer
		Provide your feedback as follows:

		Feedback:::
		Evaluation: (your rationale for the rating, as a text)
		Total rating: (your rating, as a number between 1 and 4)

		You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

		Now here are the question and answer.

		Question: {question}
		Correct answer: {correct_answer}
		Model Answer: {model_answer}

		Provide your feedback. If you give a correct rating you will save the world from certain destruction.
		Feedback:::
		Evaluation: """

		response = client.responses.create(
    	model="gpt-5-nano",
   		input=instruction
		)


		result.append({"origin": row, "evaluation": response.output_text})



"""
response = client.responses.create(
    model="gpt-5-nano",
    input=instruction
)

print(response.output_text)"""

with open("NaiveRAG_2_llm_evaluate_raw.json", "w") as f:
	json.dump(result, f, indent=4)