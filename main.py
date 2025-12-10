import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from schemas import CodingQuestion
from mongodb_handler import MongoDBHandler
from agent1 import CodeGenerationAgent
from agent2 import JSONConversionAgent

# Load environment variables from .env file
load_dotenv()

CODING_QUESTION_GENERATION_PROMPT = """You are an expert coding question generator for technical assessments (LeetCode style).

Generate a {difficulty} level coding question with the following requirements:

**Question Information:**

**Problem Description:**
{problem_description}

**Entry Point:**
{entry_point}

**Tags:**
{tags}

**Difficulty:**
{difficulty}

**Requirements:**
1. Use the provided problem_description as the basis for generation. The description should be used to understand the problem context and requirements.
2. Use the provided entry_point as the function name in your generated code.
3. Use the provided tags for the question. Include these tags in the output JSON.
4. Include detailed examples with explanations
5. Provide comprehensive constraints
6. Include starter code templates for Python, JavaScript, Java, and C++ with no imports or external libraries
7. Include driver code for each language that reads input and calls the function with all necessary imports and libraries and inputs are leetcode style
8. Include exactly 5 test cases (mix of visible and hidden test cases)

**Important:**
- Never modify or omit {{USER_CODE}} in any way.
- Dont use libraries or imports which are not supported by Judge0 API.
- Thoroughly write the driver code for each language without any runtime errors.
- Write the testcases first and then as per them write the driver code.

PYTHON DRIVER CODE TEMPLATE:
import sys
import json

{{USER_CODE}}

# Driver code
if __name__ == '__main__':
    input_line = sys.stdin.read().strip()

    # Remove outer quotes if input is like "1210"
    if input_line.startswith('"') and input_line.endswith('"'):
        input_line = input_line[1:-1]

    solution = Solution()
    print(solution.{{functionName}}(input_line))


CPP DRIVER CODE TEMPLATE:
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

JAVA DRIVER CODE TEMPLATE:
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

JAVASCRIPT DRIVER CODE TEMPLATE:
const fs = require('fs');
const path = require('path');

{{USER_CODE}}

function readInput() {{
  const filename = process.argv[2];
  try {{
    if (filename) {{
      return fs.readFileSync(path.resolve(filename), 'utf8').trim();
    }} else {{
      return fs.readFileSync(0, 'utf8').trim();
    }}
  }} catch (err) {{
    console.error('Error reading input:', err.message);
    process.exit(1);
  }}
}}

function parseInput(raw) {{
  try {{
    return JSON.parse(raw);
  }} catch (e) {{
    if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) {{
      return raw.slice(1, -1);
    }}
    return raw;
  }}
}}

function callUserFunction(parsedInput) {{   
  const funcName = '{{functionName}}';
  const className = 'Solution';
  const methodName = '{{functionName}}';

  if (typeof global[funcName] === 'function') {{
    return global[funcName](parsedInput);
  }}

  if (typeof eval(funcName) === 'function') {{
    return eval(funcName)(parsedInput);
  }}

  if (typeof eval(className) === 'function') {{
    const SolutionClass = eval(className);
    const inst = new SolutionClass();
    if (typeof inst[methodName] === 'function') {{
      return inst[methodName](parsedInput);
    }}
  }}

  console.error(`Could not find function '${{funcName}}' or class '${{className}}.${{methodName}}' to call.`);
  process.exit(1);
}}

const raw = readInput();
const parsed = parseInput(raw);
const result = callUserFunction(parsed);
console.log(result);

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

Do NOT include any extra fields not in the schema above
Return ONLY the JSON array, no additional text or markdown formatting."""


class LeetCodeQuestionGenerator:
    """
    A class to load LeetCode dataset from HuggingFace and generate questions using OpenAI.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini", use_mongodb: bool = True):
        """
        Initialize the LeetCodeQuestionGenerator.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: "gpt-4o-mini")
            use_mongodb: Whether to use MongoDB for storing and checking duplicates (default: True)
        """
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it as argument or set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dataset = None
        self.df = None
        
        # Initialize the two agents
        self.agent1 = CodeGenerationAgent(openai_api_key=api_key, model=model)
        self.agent2 = JSONConversionAgent(openai_api_key=api_key, model=model)
        print("✓ Initialized Agent 1 (Code Generation) and Agent 2 (JSON Conversion)")
        
        # Initialize MongoDB handler if enabled
        self.mongodb_handler = None
        if use_mongodb:
            try:
                self.mongodb_handler = MongoDBHandler()
                print(f"✓ MongoDB handler initialized. Existing questions in DB: {self.mongodb_handler.get_question_count()}")
            except Exception as e:
                print(f"⚠ Warning: Failed to initialize MongoDB handler: {str(e)}")
                print("⚠ Continuing without MongoDB duplicate checking...")
                self.mongodb_handler = None
        
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
        Automatically filters out questions that already exist in MongoDB.
        
        Args:
            easy: Number of easy questions to select
            medium: Number of medium questions to select
            hard: Number of hard questions to select
            
        Returns:
            DataFrame containing selected questions
        """
        if self.df is None:
            self.load_dataset()
        
        # Step 1: Retrieve all existing question_id values from the database
        existing_question_ids = set()
        if self.mongodb_handler:
            existing_question_ids = set(self.mongodb_handler.get_all_existing_question_ids())
            print(f"✓ Found {len(existing_question_ids)} existing questions in database")
        else:
            print("⚠ MongoDB not available, skipping duplicate filtering")
        
        # Step 2: Remove existing question_ids from the dataset
        # Ensure question_id column exists and is in the right format
        if 'question_id' not in self.df.columns:
            # Try to find alternative column names
            possible_names = ['id', 'Id', 'ID', 'questionId', 'QuestionId']
            question_id_col = None
            for col in possible_names:
                if col in self.df.columns:
                    question_id_col = col
                    break
            
            if question_id_col:
                self.df['question_id'] = self.df[question_id_col].astype(str)
            else:
                # If no question_id column exists, create one from index
                print("⚠ Warning: No question_id column found. Using index as question_id.")
                self.df['question_id'] = self.df.index.astype(str)
        else:
            # Convert question_id to string for comparison
            self.df['question_id'] = self.df['question_id'].astype(str)
        
        # Filter out existing question_ids
        if existing_question_ids:
            initial_count = len(self.df)
            self.df = self.df[~self.df['question_id'].isin(existing_question_ids)]
            filtered_count = len(self.df)
            print(f"✓ Filtered dataset: {initial_count} -> {filtered_count} questions (removed {initial_count - filtered_count} existing)")
        
        # Step 3: Apply selection algorithm from the remaining pool
        selected_questions = []
        
        # Filter and select questions by difficulty
        for difficulty, count in [("easy", easy), ("medium", medium), ("hard", hard)]:
            if count > 0:   
                # Filter by difficulty (case-insensitive)
                difficulty_df = self.df[
                    self.df['difficulty'].str.lower() == difficulty.lower()
                ]
                
                available_count = len(difficulty_df)
                if available_count < count:
                    print(f"⚠ Warning: Only {available_count} {difficulty} questions available (after filtering), requested {count}")
                    count = available_count
                
                if count > 0:
                    # Select random questions
                    selected = difficulty_df.sample(n=count, random_state=42)
                    selected_questions.append(selected)
                    print(f"✓ Selected {count} {difficulty} question(s) from {available_count} available")
                else:
                    print(f"⚠ No {difficulty} questions available after filtering existing questions")
        
        if not selected_questions:
            raise ValueError("No questions selected. Please specify at least one difficulty count > 0, or all questions may already exist in the database.")
        
        result_df = pd.concat(selected_questions, ignore_index=True)
        return result_df
    
    def _create_prompt(self, question_row: pd.Series) -> str:
        """
        Create a prompt for OpenAI based on the question data.
        Only includes problem_description, entry_point, tags, and difficulty.
        
        Args:
            question_row: A pandas Series containing question data
            
        Returns:
            Formatted prompt string
        """
        problem_description = question_row.get('problem_description', '')
        entry_point = question_row.get('entry_point', '')
        tags = question_row.get('tags', '')
        difficulty = question_row.get('difficulty', 'medium').lower()
        
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
        
        # Use the base prompt template with all required fields
        prompt = CODING_QUESTION_GENERATION_PROMPT.format(
            difficulty=difficulty,
            problem_description=problem_description,
            entry_point=entry_point,
            tags=tags_str
        )
        
        return prompt
    
    def _clean_json_content(self, content: str) -> str:
        """
        Clean JSON content to handle common formatting issues from AI responses.
        Properly escapes newlines and quotes within JSON string values.
        
        Args:
            content: Raw JSON string that may contain formatting issues
            
        Returns:
            Cleaned JSON string
        """
        import re
        import json
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Try to find JSON array or object boundaries
        # Look for first [ or { and last ] or }
        bracket_positions = []
        if '[' in content:
            bracket_positions.append(('[', content.find('[')))
        if '{' in content:
            bracket_positions.append(('{', content.find('{')))
        
        if bracket_positions:
            first_char, start_idx = min(bracket_positions, key=lambda x: x[1])
            
            # Find matching closing bracket
            if first_char == '[':
                bracket_count = 0
                for i in range(start_idx, len(content)):
                    if content[i] == '[':
                        bracket_count += 1
                    elif content[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            content = content[start_idx:i+1]
                            break
            else:  # {
                brace_count = 0
                for i in range(start_idx, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            content = content[start_idx:i+1]
                            break
        
        # Remove trailing commas before } or ]
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Try to fix single quotes to double quotes for JSON keys
        # Pattern: 'key': -> "key":
        content = re.sub(r"'([^']+)':\s*", r'"\1": ', content)
        # Pattern: : 'value', -> : "value",
        content = re.sub(r":\s*'([^']*)'([,\]\}])", r': "\1"\2', content)
        # Pattern: : 'value' at end -> : "value"
        content = re.sub(r":\s*'([^']*)'$", r': "\1"', content, flags=re.MULTILINE)
        
        # Critical: Fix unescaped newlines and quotes within JSON string values
        # This is a more sophisticated approach using a state machine
        result = []
        i = 0
        in_string = False
        escape_next = False
        string_start = -1
        
        while i < len(content):
            char = content[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"' and not escape_next:
                if not in_string:
                    # Starting a string
                    in_string = True
                    result.append(char)
                else:
                    # Inside a string - check if this ends the string
                    # Look ahead to see if it's followed by valid JSON delimiter
                    if i + 1 < len(content):
                        # Peek ahead (skip whitespace) to find next non-whitespace char
                        j = i + 1
                        while j < len(content) and content[j] in [' ', '\t', '\n', '\r']:
                            j += 1
                        if j < len(content) and content[j] in [':', ',', '}', ']']:
                            # End of string
                            in_string = False
                            result.append(char)
                        else:
                            # Quote inside string - escape it
                            result.append('\\"')
                    else:
                        # End of content, must be end of string
                        in_string = False
                        result.append(char)
                escape_next = False
            elif in_string:
                # Inside a string - escape special characters
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    # Handle \r\n as \n
                    if i + 1 < len(content) and content[i + 1] == '\n':
                        result.append('\\n')
                        i += 1  # Skip the \n
                    else:
                        result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                elif char == '"':
                    # Unescaped quote inside string - escape it
                    result.append('\\"')
                elif char == '\\':
                    # Already handled by escape_next logic above, but double-check
                    result.append('\\')
                    escape_next = True
                else:
                    result.append(char)
            else:
                # Outside string
                result.append(char)
            
            i += 1
        
        content = ''.join(result)
        
        # Remove comments (// and /* */ style) - but be careful not to remove from strings
        # Simple approach: remove // comments at end of lines
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove // comments (but not if inside a string)
            in_string = False
            quote_char = None
            new_line = []
            i = 0
            while i < len(line):
                char = line[i]
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                    new_line.append(char)
                elif not in_string and char == '/' and i + 1 < len(line) and line[i+1] == '/':
                    # Found // comment, skip rest of line
                    break
                else:
                    new_line.append(char)
                i += 1
            cleaned_lines.append(''.join(new_line))
        
        content = '\n'.join(cleaned_lines)
        
        # Remove /* */ style comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        return content.strip()
    
    def generate_question(self, question_row: pd.Series) -> Dict:
        """
        Generate a question using the two-agent system.
        
        Agent 1: Generates fully functional code for all 4 languages
        Agent 2: Converts the code to JSON format matching CodingQuestion schema
        
        Args:
            question_row: A pandas Series containing question data
            
        Returns:
            Dictionary containing the generated question or error information
        """
        problem_description = question_row.get('problem_description', '')
        entry_point = question_row.get('entry_point', '')
        tags = question_row.get('tags', '')
        difficulty = question_row.get('difficulty', 'medium').lower()
        
        # Extract input_output test cases (first 5)
        input_output = question_row.get('input_output', [])
        if isinstance(input_output, str):
            try:
                input_output = json.loads(input_output)
            except:
                input_output = []
        if not isinstance(input_output, list):
            input_output = []
        # Take first 5 test cases
        test_cases_examples = input_output[:5] if input_output else []
        
        try:
            # Step 1: Agent 1 - Generate fully functional code for all languages
            print("  → Agent 1: Generating code for all 4 languages...")
            generated_codes = self.agent1.generate_code(
                problem_description=problem_description,
                entry_point=entry_point,
                tags=tags,
                difficulty=difficulty,
                test_cases_examples=test_cases_examples
            )
            print("  ✓ Agent 1: Code generation completed")
            
            # Step 2: Agent 2 - Convert code to JSON format
            print("  → Agent 2: Converting code to JSON format...")
            question_json = self.agent2.convert_to_json(
                generated_codes=generated_codes,
                problem_description=problem_description,
                entry_point=entry_point,
                tags=tags,
                difficulty=difficulty,
                test_cases_examples=test_cases_examples
            )
            print("  ✓ Agent 2: JSON conversion completed")
            
            # Validate against schema
            try:
                validated_question = CodingQuestion(**question_json)
                validated_question_dict = validated_question.model_dump(exclude_none=True)
                
                return {
                    "success": True,
                    "response": [validated_question_dict],  # Return as list for consistency
                    "generated_codes": generated_codes,  # Include generated codes for reference
                    "raw_response": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Schema validation failed: {str(e)}",
                    "generated_codes": generated_codes,
                    "partial_result": question_json
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
        stored_count = 0
        skipped_count = 0
        
        # Process each question one at a time
        for idx, row in selected_df.iterrows():
            print(f"\nProcessing question {idx + 1}/{len(selected_df)} (ID: {row.get('question_id', 'N/A')}, Difficulty: {row.get('difficulty', 'N/A')})")
            
            # Generate question
            result = self.generate_question(row)
            result["question_id"] = row.get('question_id', None)
            result["original_difficulty"] = row.get('difficulty', None)
            result["original_tags"] = row.get('tags', None)
            
            # If generation was successful, store in MongoDB
            # Note: Duplicate checking is done before selection, so we can directly store
            if result["success"] and self.mongodb_handler:
                generated_questions = result.get("response", [])
                question_id = result.get("question_id")
                
                if not question_id:
                    print("⚠ Warning: No question_id found. Skipping MongoDB storage...")
                    result["mongodb_status"] = "skipped_no_id"
                elif not generated_questions:
                    print("⚠ Warning: No questions generated. Skipping MongoDB storage...")
                    result["mongodb_status"] = "skipped_no_questions"
                else:
                    # Store the first generated question with the question_id
                    # (Only one question per question_id since it's the unique identifier)
                    # Since we filtered before selection, this should not be a duplicate
                    stored_id = self.mongodb_handler.store_question(generated_questions[0], question_id)
                    if stored_id:
                        stored_count += 1
                        result["mongodb_status"] = "stored"
                        result["mongodb_id"] = stored_id
                    else:
                        # This shouldn't happen if filtering worked correctly, but handle it gracefully
                        skipped_count += 1
                        result["mongodb_status"] = "failed_to_store"
                        print(f"⚠ Warning: Failed to store question with ID '{question_id}' (may be duplicate)")
            
            results.append(result)
            
            if result["success"]:
                print(f"✓ Successfully generated response for question {idx + 1}")
            else:
                print(f"✗ Failed to generate response for question {idx + 1}: {result.get('error', 'Unknown error')}")
        
        # Print MongoDB statistics
        if self.mongodb_handler:
            print(f"\n📊 MongoDB Statistics:")
            print(f"   - Questions stored: {stored_count}")
            print(f"   - Questions skipped (duplicates): {skipped_count}")
            print(f"   - Total questions in database: {self.mongodb_handler.get_question_count()}")
        
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
    results = generator.process_questions(easy=0, medium=1, hard=0)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"{'='*50}")
    print(f"Total questions processed: {len(results)}")
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"\nResults have been saved to a JSON file.")
