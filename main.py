import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from schemas import CodingQuestion

# Load environment variables from .env file
load_dotenv()

CODING_QUESTION_GENERATION_PROMPT = """You are an expert coding question generator for technical assessments (LeetCode style).

Generate {count} {difficulty} level coding question(s) with the following requirements:

**Requirements:**

1. Use the provided question_description as the basis for generation. The description should be used to understand the problem context and requirements.
2. Use the provided tags for the question. Include these tags in the output JSON.
3. Each question should be a classic algorithmic or data structure problem
4. Questions should be well-defined with clear problem statements
5. Include detailed examples with explanations
6. Provide comprehensive constraints
7. Include starter code templates for Python, JavaScript, Java, and C++ with no imports or external libraries
8. Include driver code for each language that reads input and calls the function with all necessary imports and libraries and inputs are leetcode style
9. Include exactly 5 test cases (mix of visible and hidden test cases)
10. Function name should be descriptive (e.g., search, twoSum, reverseString)

**Important:**
- Use the question_description provided in the context to generate required fields in the output JSON
- Use the tags provided in the context for the tags field in the output JSON
- Dont use libraries or imports which are not supported by Judge0 API.
EXAMPLE DRIVER CODE FOR CPP
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

{{USER_CODE}}

int main() {{
    std::string input_line;
    std::getline(std::cin, input_line);

    // --- FIX START: Handle brackets and commas ---
    // Turn "[1, 5, 2]" into " 1  5  2 "
    for (char &c : input_line) {{
        if (c == '[' || c == ']' || c == ',') {{
            c = ' '; 
        }}
    }}
    // --- FIX END ---

    std::istringstream iss(input_line);
    std::vector<int> arr;
    int number;
    while (iss >> number) {{
        arr.push_back(number);
    }}

    Solution solution;
    std::cout << solution.findMax(arr) << std::endl;
    return 0;
}}

Example Driver Code for Java
import java.util.*;
import java.io.*;

{{USER_CODE}}

public class Main {{
    public static void main(String[] args) {{
        Scanner scanner = new Scanner(System.in);
        if (scanner.hasNextLine()) {{
            String input = scanner.nextLine();
            
            // Fix: Remove brackets '[' and ']' before splitting
            input = input.replace("[", "").replace("]", "");
            
            // Fix: Split by comma and trim whitespace
            if (!input.trim().isEmpty()) {{
                int[] arr = Arrays.stream(input.split(","))
                                  .map(String::trim) // Remove spaces like " 5"
                                  .mapToInt(Integer::parseInt)
                                  .toArray();
                
                Solution solution = new Solution();
                System.out.println(solution.findMax(arr));
            }} else {{
                // Handle empty input case (e.g. "[]")
                System.out.println(0); // or handle appropriate empty return
            }}
        }}
    }}
}}

**Output Format:**

Return a JSON array of questions. Each question MUST strictly follow this exact structure (matching the CodingQuestion schema):

{{
  "title": "Question Title",
  "description": "Detailed problem description",
  "difficulty": "{difficulty}",
  "tags": ["tag1", "tag2"],
  "functionName": "functionName",
  "functionSignature": "def functionName(params: Type) -> ReturnType:",
  "examples": [
    {{
      "input": "input description",
      "output": "output description",
      "explanation": "explanation text (optional)"
    }}
  ],
  "constraints": [
    "constraint 1",
    "constraint 2"
  ],
  "starterCode": {{
    "python": "def functionName(params):\\n    # Your code here\\n    pass",
    "javascript": "function functionName(params) {{\\n    // Your code here\\n}}",
    "java": "class Solution {{\\n    public ReturnType functionName(Params params) {{\\n        // Your code here\\n    }}\\n}}",
    "cpp": "class Solution {{\\npublic:\\n    ReturnType functionName(Params params) {{\\n        // Your code here\\n    }}\\n}};"
  }},
  "driverCode": {{
    "python": "import sys\\nimport json\\n\\n{{USER_CODE}}\\n\\n# Driver code\\n# ...",
    "javascript": "import fs from 'fs';\\nimport path from 'path';\\n\\n{{USER_CODE}}\\n\\n// Driver code\\n// ...",
    "java": "import java.util.*;\\nimport java.io.*;\\n{{USER_CODE}}\\n\\npublic class Main {{\\n    // Driver code\\n}}",
    "cpp": "import <iostream>\\n#include <vector>\\n#include <string>\\n#include <sstream>\\nusing namespace std;\\n\\n{{USER_CODE}}\\n\\nint main() {{\\n    // Driver code\\n}}"
  }},
  "testCases": [
    {{
      "input": "test input string",
      "expectedOutput": "expected output",
      "isHidden": false
    }},
    {{
      "input": "test input string",
      "expectedOutput": "expected output",
      "isHidden": true
    }}
  ]
}}

**CRITICAL REQUIREMENTS:**
- Include exactly 5 test cases in the testCases array
- All field names must match exactly (case-sensitive): title, description, difficulty, tags, functionName, functionSignature, examples, constraints, starterCode, driverCode, testCases
- examples array items must have: input (required), output (required), explanation (optional)
- testCases array items must have: input (required), expectedOutput (required), isHidden (required, boolean)
- starterCode must have: python, javascript, java, cpp (all required)
- driverCode must have: python, javascript, java, cpp (all required)
- Do NOT include any extra fields not in the schema above
- Do NOT include _id or id fields

Return ONLY the JSON array, no additional text or markdown formatting."""


class LeetCodeQuestionGenerator:
    """
    A class to load LeetCode dataset from HuggingFace and generate questions using OpenAI.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the LeetCodeQuestionGenerator.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: "gpt-4o-mini")
        """
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it as argument or set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dataset = None
        self.df = None
        
    def load_dataset(self):
        """
        Load the LeetCode dataset from HuggingFace.
        """
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("newfacade/LeetCodeDataset")
        
        # Convert to pandas DataFrame for easier manipulation
        if isinstance(dataset, dict):
            # If dataset has splits, use the first one (usually 'train')
            split_name = list(dataset.keys())[0]
            self.df = dataset[split_name].to_pandas()
        else:
            self.df = dataset.to_pandas()
        
        print(f"Dataset loaded successfully. Total questions: {len(self.df)}")
        return self.df
    
    def select_questions(self, easy: int = 0, medium: int = 0, hard: int = 0) -> pd.DataFrame:
        """
        Select questions from the dataset based on difficulty counts.
        
        Args:
            easy: Number of easy questions to select
            medium: Number of medium questions to select
            hard: Number of hard questions to select
            
        Returns:
            DataFrame containing selected questions
        """
        if self.df is None:
            self.load_dataset()
        
        selected_questions = []
        
        # Filter and select questions by difficulty
        for difficulty, count in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
            if count > 0:
                # Filter by difficulty (case-insensitive)
                difficulty_df = self.df[
                    self.df['difficulty'].str.lower() == difficulty.lower()
                ]
                
                if len(difficulty_df) < count:
                    print(f"Warning: Only {len(difficulty_df)} {difficulty} questions available, requested {count}")
                    count = len(difficulty_df)
                
                # Select random questions
                selected = difficulty_df.sample(n=count, random_state=42)
                selected_questions.append(selected)
                print(f"Selected {count} {difficulty} question(s)")
        
        if not selected_questions:
            raise ValueError("No questions selected. Please specify at least one difficulty count > 0")
        
        result_df = pd.concat(selected_questions, ignore_index=True)
        return result_df
    
    def _create_prompt(self, question_row: pd.Series) -> str:
        """
        Create a prompt for OpenAI based on the question data.
        
        Args:
            question_row: A pandas Series containing question data
            
        Returns:
            Formatted prompt string
        """
        problem_description = question_row.get('problem_description', '')
        starter_code = question_row.get('starter_code', '')
        entry_point = question_row.get('entry_point', '')
        tags = question_row.get('tags', '')
        difficulty = question_row.get('difficulty', 'Medium')
        
        # Parse tags if it's a string (could be JSON or comma-separated)
        if isinstance(tags, str):
            try:
                tags_list = json.loads(tags)
                if isinstance(tags_list, list):
                    tags_str = ", ".join(tags_list)
                else:
                    tags_str = tags
            except:
                tags_str = tags
        else:
            tags_str = str(tags)
        
        # Create context from the question
        context = f"""Based on the following LeetCode question, generate the same question:

**Question Description (use this as the basis for your question):**
{problem_description}

**Tags (use these tags in your output):**
{tags_str}

**Starter Code:**
{starter_code}

**Entry Point:**
{entry_point}

**Difficulty:**
{difficulty}

"""
        
        # Use the base prompt template
        prompt = CODING_QUESTION_GENERATION_PROMPT.format(
            count=1,
            difficulty=difficulty
        )
        
        # Combine context with the prompt
        full_prompt = f"{context}\n\n{prompt}"
        
        return full_prompt
    
    def generate_question(self, question_row: pd.Series) -> Dict:
        """
        Generate a question using OpenAI API for a single question row.
        
        Args:
            question_row: A pandas Series containing question data
            
        Returns:
            Dictionary containing the OpenAI response
        """
        prompt = self._create_prompt(question_row)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert coding question generator. Always return valid JSON arrays that strictly match the CodingQuestion schema. Ensure all required fields are present and field names match exactly (case-sensitive)."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Parse JSON - should be an array
                result = json.loads(content)
                
                # Ensure it's a list/array
                if not isinstance(result, list):
                    # If it's a dict with a key containing the array, extract it
                    if isinstance(result, dict) and len(result) == 1:
                        result = list(result.values())[0]
                    else:
                        # Wrap in array if single object
                        result = [result]
                
                # Validate each question against the schema
                validated_questions = []
                validation_errors = []
                
                for idx, question_data in enumerate(result):
                    try:
                        # Validate against CodingQuestion schema
                        validated_question = CodingQuestion(**question_data)
                        # Convert back to dict for JSON serialization
                        validated_questions.append(validated_question.model_dump(exclude_none=True))
                    except Exception as e:
                        validation_errors.append(f"Question {idx + 1}: {str(e)}")
                
                if validation_errors:
                    return {
                        "success": False,
                        "error": f"Schema validation failed: {'; '.join(validation_errors)}",
                        "raw_response": content,
                        "partial_result": validated_questions if validated_questions else None
                    }
                
                return {
                    "success": True,
                    "response": validated_questions,
                    "raw_response": content
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse JSON: {str(e)}",
                    "raw_response": content
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_response": None
            }
    
    def process_questions(self, easy: int = 0, medium: int = 0, hard: int = 0, output_file: Optional[str] = None) -> List[Dict]:
        """
        Process questions: select from dataset and generate responses for each.
        
        Args:
            easy: Number of easy questions to process
            medium: Number of medium questions to process
            hard: Number of hard questions to process
            output_file: Optional path to save results as JSON file. If None, defaults to 'results_<timestamp>.json'
            
        Returns:
            List of dictionaries containing results for each question
        """
        # Select questions
        selected_df = self.select_questions(easy=easy, medium=medium, hard=hard)
        
        results = []
        
        # Process each question one at a time
        for idx, row in selected_df.iterrows():
            print(f"\nProcessing question {idx + 1}/{len(selected_df)} (ID: {row.get('question_id', 'N/A')}, Difficulty: {row.get('difficulty', 'N/A')})")
            
            result = self.generate_question(row)
            result["question_id"] = row.get('question_id', None)
            result["original_difficulty"] = row.get('difficulty', None)
            result["original_tags"] = row.get('tags', None)
            
            results.append(result)
            
            if result["success"]:
                print(f"✓ Successfully generated response for question {idx + 1}")
            else:
                print(f"✗ Failed to generate response for question {idx + 1}: {result.get('error', 'Unknown error')}")
        
        # Save results to JSON file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results_{timestamp}.json"
        
        try:
            # Helper function to convert numpy/pandas types to native Python types
            def convert_to_serializable(obj):
                """Recursively convert numpy/pandas types to native Python types"""
                import numpy as np
                
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {str(key): convert_to_serializable(value) for key, value in obj.items()}
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            # Convert results to serializable format
            serializable_results = convert_to_serializable(results)
            
            # Prepare data for JSON serialization
            json_data = {
                "metadata": {
                    "total_questions": len(results),
                    "easy_count": easy,
                    "medium_count": medium,
                    "hard_count": hard,
                    "generated_at": datetime.now().isoformat(),
                    "model": self.model
                },
                "results": serializable_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to {output_file}")
        except Exception as e:
            print(f"\n✗ Failed to save results to file: {str(e)}")
        
        return results


if __name__ == "__main__":
    # Example usage
    generator = LeetCodeQuestionGenerator()
    
    # Process questions: 2 easy, 2 medium, 1 hard
    # Results will be automatically saved to a JSON file (default: results_<timestamp>.json)
    # You can also specify a custom filename: generator.process_questions(easy=2, medium=2, hard=1, output_file="my_results.json")
    results = generator.process_questions(easy=2, medium=2, hard=1)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"{'='*50}")
    print(f"Total questions processed: {len(results)}")
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"\nResults have been saved to a JSON file.")
