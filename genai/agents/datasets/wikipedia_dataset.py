import mlflow
from mlflow.genai.datasets import create_dataset
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Wikipedia Knowledge Evaluation")

# Create dataset from current experiment
dataset = create_dataset(
    name="wikipedia_knowledge_dataset",
    experiment_id=mlflow.get_experiment_by_name("Wikipedia Knowledge Evaluation").experiment_id,
    tags={"stage": "validation", "domain": "wikipedia", "size": "20"},
)

# Create Wikipedia test cases covering various topics
dataset_wikipedia = pd.DataFrame(
    [
        {
            "inputs": {
                "question": "What is the capital of France?",
                "domain": "geography",
            },
            "expectations": {
                "expected_answer": "Paris is the capital and largest city of France",
                "must_mention": ["Paris"],
            },
        },
        {
            "inputs": {
                "question": "Who painted the Mona Lisa?",
                "domain": "art",
            },
            "expectations": {
                "expected_answer": "Leonardo da Vinci painted the Mona Lisa",
                "must_mention": ["Leonardo da Vinci", "Renaissance"],
            },
        },
        {
            "inputs": {
                "question": "When did World War II end?",
                "domain": "history",
            },
            "expectations": {
                "expected_answer": "World War II ended in 1945",
                "must_mention": ["1945"],
            },
        },
        {
            "inputs": {
                "question": "What is photosynthesis?",
                "domain": "science",
            },
            "expectations": {
                "expected_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy",
                "must_mention": ["plants", "light", "energy", "chlorophyll"],
            },
        },
        {
            "inputs": {
                "question": "Who wrote Romeo and Juliet?",
                "domain": "literature",
            },
            "expectations": {
                "expected_answer": "William Shakespeare wrote Romeo and Juliet",
                "must_mention": ["William Shakespeare", "playwright"],
            },
        },
        {
            "inputs": {
                "question": "What is the largest planet in our solar system?",
                "domain": "astronomy",
            },
            "expectations": {
                "expected_answer": "Jupiter is the largest planet in our solar system",
                "must_mention": ["Jupiter", "gas giant"],
            },
        },
        {
            "inputs": {
                "question": "What is DNA?",
                "domain": "biology",
            },
            "expectations": {
                "expected_answer": "DNA (deoxyribonucleic acid) is a molecule that contains genetic instructions",
                "must_mention": ["genetic", "molecule", "heredity"],
            },
        },
        {
            "inputs": {
                "question": "Who was the first president of the United States?",
                "domain": "history",
            },
            "expectations": {
                "expected_answer": "George Washington was the first president of the United States",
                "must_mention": ["George Washington"],
            },
        },
        {
            "inputs": {
                "question": "What is the speed of light?",
                "domain": "physics",
            },
            "expectations": {
                "expected_answer": "The speed of light is approximately 299,792,458 meters per second",
                "must_mention": ["299,792", "meters per second", "vacuum"],
            },
        },
        {
            "inputs": {
                "question": "What is the Great Wall of China?",
                "domain": "architecture",
            },
            "expectations": {
                "expected_answer": "The Great Wall of China is an ancient fortification built to protect Chinese states",
                "must_mention": ["fortification", "China", "defense"],
            },
        },
        {
            "inputs": {
                "question": "Who discovered penicillin?",
                "domain": "medicine",
            },
            "expectations": {
                "expected_answer": "Alexander Fleming discovered penicillin in 1928",
                "must_mention": ["Alexander Fleming", "antibiotic", "1928"],
            },
        },
        {
            "inputs": {
                "question": "What is the theory of relativity?",
                "domain": "physics",
            },
            "expectations": {
                "expected_answer": "Einstein's theory of relativity describes the relationship between space and time",
                "must_mention": ["Einstein", "space", "time", "gravity"],
            },
        },
        {
            "inputs": {
                "question": "What is the Pythagorean theorem?",
                "domain": "mathematics",
            },
            "expectations": {
                "expected_answer": "The Pythagorean theorem states that in a right triangle, a² + b² = c²",
                "must_mention": ["right triangle", "hypotenuse", "squared"],
            },
        },
        {
            "inputs": {
                "question": "Who composed the Fifth Symphony?",
                "domain": "music",
            },
            "expectations": {
                "expected_answer": "Ludwig van Beethoven composed the Fifth Symphony",
                "must_mention": ["Beethoven", "classical music"],
            },
        },
        {
            "inputs": {
                "question": "What is the Amazon rainforest?",
                "domain": "ecology",
            },
            "expectations": {
                "expected_answer": "The Amazon rainforest is the world's largest tropical rainforest in South America",
                "must_mention": ["rainforest", "South America", "biodiversity"],
            },
        },
        {
            "inputs": {
                "question": "What caused the extinction of dinosaurs?",
                "domain": "paleontology",
            },
            "expectations": {
                "expected_answer": "A massive asteroid impact about 66 million years ago likely caused dinosaur extinction",
                "must_mention": ["asteroid", "extinction", "Cretaceous"],
            },
        },
        {
            "inputs": {
                "question": "What is the Eiffel Tower?",
                "domain": "architecture",
            },
            "expectations": {
                "expected_answer": "The Eiffel Tower is an iron lattice tower in Paris built by Gustave Eiffel",
                "must_mention": ["Paris", "Gustave Eiffel", "iron"],
            },
        },
        {
            "inputs": {
                "question": "Who invented the telephone?",
                "domain": "technology",
            },
            "expectations": {
                "expected_answer": "Alexander Graham Bell is credited with inventing the telephone",
                "must_mention": ["Alexander Graham Bell", "communication"],
            },
        },
        {
            "inputs": {
                "question": "What is the periodic table?",
                "domain": "chemistry",
            },
            "expectations": {
                "expected_answer": "The periodic table organizes chemical elements by atomic number and properties",
                "must_mention": ["elements", "atomic number", "Mendeleev"],
            },
        },
        {
            "inputs": {
                "question": "What is the Magna Carta?",
                "domain": "history",
            },
            "expectations": {
                "expected_answer": "The Magna Carta is a 1215 charter that limited royal power in England",
                "must_mention": ["1215", "England", "rights", "King John"],
            },
        },
    ]
)

# Merge records into the dataset
dataset.merge_records(dataset_wikipedia)

print(f"Created dataset with {len(dataset_wikipedia)} Wikipedia test cases")
print(f"Dataset name: {dataset.name}")
print("Experiment: Wikipedia Knowledge Evaluation")
