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
        for difficulty, count in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
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
                
                # Try parsing first - if it works, great!
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # If parsing fails, try cleaning and parsing again
                    content = self._clean_json_content(content)
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
                # Try to extract more context about the error
                error_msg = f"Failed to parse JSON: {str(e)}"
                if hasattr(e, 'pos') and e.pos is not None:
                    # Show context around the error
                    start = max(0, e.pos - 100)
                    end = min(len(content), e.pos + 100)
                    error_context = content[start:end]
                    error_msg += f"\nError context (char {e.pos}): ...{error_context}..."
                
                # Save raw response for debugging
                debug_file = f"debug_json_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                try:
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write("Raw response from OpenAI:\n")
                        f.write("=" * 80 + "\n")
                        f.write(response.choices[0].message.content)
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("\nCleaned content:\n")
                        f.write("=" * 80 + "\n")
                        f.write(content)
                    print(f"⚠ Debug info saved to {debug_file}")
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "error": error_msg,
                    "raw_response": content[:500] if len(content) > 500 else content  # Truncate for response
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
