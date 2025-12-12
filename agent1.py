"""
Agent 1: Code Generation Agent
Generates fully functional, runnable code for Python, JavaScript, Java, and C++
based on problem_description, entry_point, and question context.
"""

import os
import json
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class CodeGenerationAgent:
    """
    Agent responsible for generating fully functional code for all 4 languages.
    Generates code that is immediately runnable and compilable without modifications.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the Code Generation Agent.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: "gpt-4o-mini")
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it as argument or set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_code(self, problem_description: str, entry_point: str, tags: str, difficulty: str, test_cases_examples: list = None) -> Dict[str, str]:
        """
        Generate fully functional code for all 4 languages.
        
        Args:
            problem_description: The problem description
            entry_point: The function/class name to implement
            tags: Tags for the problem
            difficulty: Difficulty level (easy, medium, hard)
            test_cases_examples: List of test cases from dataset (first 5) showing input/output format
            
        Returns:
            Dictionary with keys: 'python', 'javascript', 'java', 'cpp'
            Each value is a complete, runnable code string
        """
        if test_cases_examples is None:
            test_cases_examples = []
        prompt = self._create_code_generation_prompt(problem_description, entry_point, tags, difficulty, test_cases_examples)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert code generator. Generate fully functional, runnable code for Python, JavaScript, Java, and C++.
                        
Your response MUST be a JSON object with exactly 4 keys: "python", "javascript", "java", "cpp".
Each value should be a complete, runnable code string that:
1. Includes all necessary imports
2. Contains the solution class/function with the exact name specified in entry_point
3. Includes driver code that reads input and calls the function for Judge0 testing
4. Is immediately compilable and runnable without any modifications
5. Output format must be CONSISTENT across all languages - use lowercase "true"/"false" for booleans (even in Python)

Return ONLY valid JSON, no markdown, no explanations."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent code generation
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Validate that we have all 4 languages
            required_languages = ['python', 'javascript', 'java', 'cpp']
            for lang in required_languages:
                if lang not in result:
                    raise ValueError(f"Missing code for language: {lang}")
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to generate code: {str(e)}")
    
    def _create_code_generation_prompt(self, problem_description: str, entry_point: str, tags: str, difficulty: str, test_cases_examples: list = None) -> str:
        """
        Create the prompt for code generation.
        
        Args:
            problem_description: The problem description
            entry_point: The function/class name to implement
            tags: Tags for the problem
            difficulty: Difficulty level
            test_cases_examples: List of test cases showing input/output format
            
        Returns:
            Formatted prompt string
        """
        if test_cases_examples is None:
            test_cases_examples = []
        
        # Check if any test case has boolean output (true/false)
        has_boolean_output = False
        if test_cases_examples:
            for test_case in test_cases_examples[:5]:
                output_val = ""
                if isinstance(test_case, dict):
                    output_val = str(test_case.get('output', '')).lower()
                elif isinstance(test_case, list) and len(test_case) >= 2:
                    output_val = str(test_case[1]).lower()
                
                if "true" in output_val or "false" in output_val:
                    has_boolean_output = True
                    break
        
        # Format test cases examples
        test_cases_section = ""
        if test_cases_examples:
            test_cases_section = "\n**Test Cases Examples (Input/Output Format from Dataset):**\n"
            test_cases_section += "These examples show the EXACT format of input (as string) and expected output (as string) from the LeetCode dataset:\n"
            test_cases_section += "The input string format is like: 'nums = [3,3], target = 6' (variable assignments as a string)\n"
            test_cases_section += "The output string format is like: '[0, 1]' or 'None' (string representation of the result)\n\n"
            
            for idx, test_case in enumerate(test_cases_examples[:5], 1):
                if isinstance(test_case, dict):
                    input_val = test_case.get('input', '')
                    output_val = test_case.get('output', '')
                    test_cases_section += f"Test Case {idx}:\n"
                    test_cases_section += f"  Input (string): {input_val}\n"
                    test_cases_section += f"  Expected Output (string): {output_val}\n\n"
                elif isinstance(test_case, list) and len(test_case) >= 2:
                    test_cases_section += f"Test Case {idx}:\n"
                    input_val = test_case[0]
                    output_val = test_case[1]
                    test_cases_section += f"  Input (string): {input_val}\n"
                    test_cases_section += f"  Expected Output (string): {output_val}\n\n"
            
            test_cases_section += "**CRITICAL INPUT/OUTPUT FORMAT REQUIREMENTS:**\n"
            test_cases_section += "1. Input Format: The stdin will be a STRING in the format shown above (e.g., 'nums = [3,3], target = 6')\n"
            test_cases_section += "   - You MUST parse this string to extract the variable values\n"
            test_cases_section += "   - The input string contains variable assignments separated by commas\n"
            test_cases_section += "   - Parse each variable assignment (e.g., 'nums = [3,3]' -> extract the array [3,3])\n"
            test_cases_section += "   - Parse numeric values, arrays, strings, etc. from the input string\n"
            test_cases_section += "2. Output Format: The output must be a STRING matching the exact format shown in examples\n"
            test_cases_section += "   - Arrays should be output as string like '[0, 1]'\n"
            test_cases_section += "   - None/null should be output as string 'None' or 'null'\n"
            test_cases_section += "   - Numbers should be output as string representation\n"
            test_cases_section += "   - Match the EXACT format from the examples (spacing, brackets, etc.)\n"
        
        # Get the conditional boolean point
        boolean_point = self._get_boolean_point(has_boolean_output)
        
        prompt = f"""Generate fully functional, runnable code for all 4 languages based on the following problem:

**Problem Description:**
{problem_description}

**Entry Point (Function/Class Name):**
{entry_point}

**Tags:**
{tags}

**Difficulty:**
{difficulty}
{test_cases_section}
**Requirements:**

1. Generate complete, runnable code for Python, JavaScript, Java, and C++

2. For each language, the code must include:
   - All necessary imports at the top
   - The solution class or function with the EXACT name: {entry_point}
   - Driver code that reads input from stdin as a STRING (Judge0 format: stdin is always a string)
   - Driver code that outputs the result as a STRING
   - The code must be immediately compilable and runnable
   - Judge0 Request Format: code (string), stdin (string), expectedOutput (string)
{boolean_point}
3. Python:
   - Use class Solution with method {entry_point} OR function {entry_point}
   - Read input from sys.stdin.read() as a STRING (not parsed)
   - The input will be a string like "nums = [3,3], target = 6" from Judge0 stdin
   - Parse the input string to extract variable values (e.g., parse "nums = [3,3], target = 6" to get nums=[3,3] and target=6)
   - Call the solution with parsed values
   - Convert the result to a STRING matching the output format (e.g., "[0, 1]" or "None")
   - IMPORTANT: Convert Python True/False to lowercase "true"/"false" strings
   - Print the result as a STRING
   - Example: 
     input_str = sys.stdin.read().strip()  # e.g., "nums = [3,3], target = 6"
     # Parse input_str to extract nums and target
     # Call solution
     # Convert result to string format matching examples
     # If result is boolean, convert True->"true", False->"false"
     result_str = str(result).replace("True", "true").replace("False", "false")
     print(result_str)

4. JavaScript:
   - Use class Solution with method {entry_point} OR function {entry_point}
   - Read input from stdin as a STRING using fs.readFileSync(0, 'utf8') or process.stdin
   - The input will be a string like "nums = [3,3], target = 6" from Judge0 stdin
   - Parse the input string to extract variable values (e.g., parse "nums = [3,3], target = 6" to get nums=[3,3] and target=6)
   - Call the solution with parsed values
   - Convert the result to a STRING matching the output format (e.g., "[0, 1]" or "None")
   - Console.log the result as a STRING
   - Example: 
     const input = fs.readFileSync(0, 'utf8').trim();  // e.g., "nums = [3,3], target = 6"
     // Parse input to extract variables
     // Call solution
     // Convert result to string format matching examples
     console.log(resultString);

5. Java:
   - Use class Solution with public method {entry_point}
   - Read input from Scanner(System.in) as a STRING using nextLine()
   - The input will be a string like "nums = [3,3], target = 6" from Judge0 stdin
   - Parse the input string to extract variable values (e.g., parse "nums = [3,3], target = 6" to get nums array and target int)
   - Call the solution with parsed values
   - Convert the result to a STRING matching the output format (e.g., "[0, 1]" or "None")
   - System.out.println the result as a STRING
   - Example: 
     String input = scanner.nextLine();  // e.g., "nums = [3,3], target = 6"
     // Parse input to extract variables
     // Call solution
     // Convert result to string format matching examples
     System.out.println(resultString);

6. C++:
   - Use class Solution with public method {entry_point}
   - Read input from std::cin as a STRING using std::getline
   - The input will be a string like "nums = [3,3], target = 6" from Judge0 stdin
   - Parse the input string to extract variable values (e.g., parse "nums = [3,3], target = 6" to get nums vector and target int)
   - Call the solution with parsed values
   - Convert the result to a STRING matching the output format (e.g., "[0, 1]" or "None")
   - std::cout the result as a STRING
   - Example: 
     std::string input;
     std::getline(std::cin, input);  // e.g., "nums = [3,3], target = 6"
     // Parse input to extract variables
     // Call solution
     // Convert result to string format matching examples
     std::cout << resultString << std::endl;

7. IMPORTANT - Judge0 Format & Input Parsing:
   - stdin is always provided as a STRING in the format: "variable1 = value1, variable2 = value2, ..."
   - Example input: "nums = [3,3], target = 6"
   - Your code MUST:
     a) Read stdin as a raw string
     b) Parse the string to extract variable names and values
     c) Handle different value types: arrays "[1,2,3]", numbers "6", strings "hello", etc.
     d) Extract each variable assignment (e.g., "nums = [3,3]" -> extract array [3,3])
     e) Pass parsed values to your solution function
     f) Convert the result to a STRING matching the output format from examples
   - Output must be a STRING matching the EXACT format from test cases examples
   - CRITICAL: Output format must be CONSISTENT across all languages:
     * Boolean values: Use lowercase "true"/"false" (NOT "True"/"False" in Python)
     * Arrays: Output as string like "[0, 1]" (with exact spacing)
     * None/null: Output as string "None" or "null" (match the format in examples)
     * Numbers: Output as string representation
   - Python-specific: Convert True/False to "true"/"false" strings to match other languages
   - Match the EXACT format shown in the test cases examples above (spacing, brackets, case, etc.)
   - All languages must produce IDENTICAL string output for the same input

8. Make sure the code is production-ready and handles edge cases.

Return a JSON object with keys: "python", "javascript", "java", "cpp"
Each value should be the complete code as a string (use \\n for newlines in JSON)."""
        
        return prompt
    
    def _get_boolean_point(self, has_boolean_output: bool) -> str:
        """
        Returns the 5th point about boolean handling if test cases contain boolean outputs.
        
        Args:
            has_boolean_output: Whether any test case has boolean output (true/false)
            
        Returns:
            String with the 5th point if has_boolean_output is True, empty string otherwise
        """
        if has_boolean_output:
            return """
5. CRITICAL - Boolean Output Format (REQUIRED for this problem):
   - The test cases contain boolean outputs (true/false)
   - ALL languages MUST output lowercase "true" or "false" as strings
   - Python-specific: You MUST convert Python's True/False to lowercase "true"/"false" strings
   - Example conversion in Python:
     result_str = str(result).replace("True", "true").replace("False", "false")
   - JavaScript, Java, and C++ should also output "true"/"false" as strings (not boolean values)
   - The output format must match EXACTLY: "true" or "false" (lowercase, as strings)
   - This is critical because test cases expect string output, not boolean values"""
        return ""

