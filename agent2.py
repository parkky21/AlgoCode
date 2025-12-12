"""
Agent 2: JSON Conversion Agent
Takes generated code from Agent 1 and converts it to the desired JSON format.
Extracts driver code, creates starter code, and generates test cases, constraints, etc.
"""

import os
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
from schemas import CodingQuestion

load_dotenv()


class JSONConversionAgent:
    """
    Agent responsible for converting generated code into the desired JSON format.
    Extracts driver code, creates starter code, and generates all required fields.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the JSON Conversion Agent.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: "gpt-4o-mini")
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it as argument or set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def convert_to_json(self, 
                       generated_codes: Dict[str, str],
                       problem_description: str,
                       entry_point: str,
                       tags: str,
                       difficulty: str,
                       test_cases_examples: list = None) -> Dict[str, Any]:
        """
        Convert generated code to JSON format matching CodingQuestion schema.
        
        Args:
            generated_codes: Dictionary with keys 'python', 'javascript', 'java', 'cpp' containing full code
            problem_description: The original problem description
            entry_point: The function/class name
            tags: Tags for the problem
            difficulty: Difficulty level
            test_cases_examples: List of test cases from dataset (first 5) showing input/output format
            
        Returns:
            Dictionary matching CodingQuestion schema
        """
        if test_cases_examples is None:
            test_cases_examples = []
        prompt = self._create_conversion_prompt(
            generated_codes, problem_description, entry_point, tags, difficulty, test_cases_examples
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at converting code into structured JSON format for coding questions.
                        
Your task is to:
1. Extract driver code from the provided full code - USE THE ENTIRE CODE, DO NOT SKIP ANY ESSENTIAL PARTS
2. Create starter code (solution class/function with placeholder for main logic only)
3. Generate test cases (5 total, mix of visible and hidden)
4. Generate constraints
5. Generate examples with explanations
6. Create proper function signatures

CRITICAL RULES:
- Driver code must include ALL essential code from the full code (imports, helper functions, classes, etc.)
- Only the main solution logic should be in starter code (the core algorithm)
- Everything else (parsing logic, helper functions, utility classes) should remain in driver code
- Do NOT skip any code that is essential for compilation/running
- Starter code should only contain the skeleton of the main solution function/class method

Return ONLY valid JSON matching the CodingQuestion schema. No markdown, no explanations."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate against schema
            try:
                validated_question = CodingQuestion(**result)
                return validated_question.model_dump(exclude_none=True)
            except Exception as e:
                raise ValueError(f"Schema validation failed: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Failed to convert to JSON: {str(e)}")
    
    def _create_conversion_prompt(self,
                                 generated_codes: Dict[str, str],
                                 problem_description: str,
                                 entry_point: str,
                                 tags: str,
                                 difficulty: str,
                                 test_cases_examples: list = None) -> str:
        """
        Create the prompt for JSON conversion.
        
        Args:
            generated_codes: Dictionary with full code for all languages
            problem_description: The problem description
            entry_point: The function/class name
            tags: Tags for the problem
            difficulty: Difficulty level
            test_cases_examples: List of test cases showing input/output format
            
        Returns:
            Formatted prompt string
        """
        if test_cases_examples is None:
            test_cases_examples = []
        # Parse tags
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
        
        # Format test cases section
        test_cases_section = self._format_test_cases_section(test_cases_examples)
        
        prompt = f"""Convert the following generated code into the proper JSON format for a coding question.

**Problem Description:**
{problem_description}

**Entry Point:**
{entry_point}

**Tags:**
{tags_str}

**Difficulty:**
{difficulty}
{test_cases_section}
**Generated Code:**

**Python:**
```python
{generated_codes.get('python', '')}
```

**JavaScript:**
```javascript
{generated_codes.get('javascript', '')}
```

**Java:**
```java
{generated_codes.get('java', '')}
```

**C++:**
```cpp
{generated_codes.get('cpp', '')}
```

**Your Task:**

1. Extract the driver code from each language's full code
   - Driver code must include EVERYTHING from the full code EXCEPT the main solution logic
   - Include ALL imports, helper functions, utility classes, parsing logic
   - Include ALL code needed for reading input, parsing, and output formatting
   - Keep the complete input parsing logic (extracting variables from string like "nums = [3,3], target = 6")
   - Keep the complete output formatting logic (converting result to string format)
   - IMPORTANT: Do NOT skip any essential code - everything except the core solution algorithm should be in driver code
   - The driver code should be fully functional and compilable when combined with starter code
   - Use {{USER_CODE}} placeholder in driver code where the starter code should be inserted
   - IMPORTANT: Ensure driver code reads stdin as a STRING and outputs as a STRING
   - Judge0 provides stdin as a string, so code should read it as raw string first, then parse if needed
   - Output must always be a string (convert arrays/objects to string representation)

2. Create starter code for each language
   - Starter code should ONLY contain the solution class/function with the name {entry_point}
   - Replace ONLY the main algorithm/logic implementation with placeholder comments
   - Keep the function signature exactly as in the full code
   - Keep method/function structure, but replace the core logic with "# Your code here" or "// Your code here"
   - DO NOT include imports, helper functions, or parsing logic in starter code
   - Starter code should be a minimal skeleton that users fill in with their solution
   - Use {{USER_CODE}} placeholder in driver code where the starter code should be inserted

3. Generate exactly 5 test cases
   - Use the test cases examples provided above as reference for input/output format
   - Input format: STRING like "nums = [3,3], target = 6" (variable assignments as a string)
   - Output format: STRING like "[0, 1]" or "None" (string representation matching examples)
   - Mix of visible (isHidden: false) and hidden (isHidden: true) test cases
   - Test cases should cover edge cases and normal scenarios
   - If examples are provided, use them directly or create similar ones following the exact format
   - Input string should follow the pattern: "variable1 = value1, variable2 = value2, ..."
   - Output string should match the exact format from examples (spacing, brackets, None vs null, etc.)

4. Generate constraints
   - Based on the problem description
   - Include input size limits, value ranges, etc.

5. Generate examples (at least 2-3)
   - Each with input, output, and explanation
   - Should help users understand the problem

6. Create function signature
   - Extract from the generated code
   - Should match the entry_point name

7. Generate title
   - A concise, descriptive title for the problem

Return a JSON object matching this exact structure:

{{
  "title": "Question Title",
  "description": "Detailed problem description based on the provided problem_description",
  "difficulty": "{difficulty}",
  "tags": ["tag1", "tag2"],
  "functionName": "{entry_point}",
  "functionSignature": "def {entry_point}(params: Type) -> ReturnType:",
  "examples": [
    {{
      "input": "input description",
      "output": "output description",
      "explanation": "explanation text"
    }}
  ],
  "constraints": [
    "constraint 1",
    "constraint 2"
  ],
  "starterCode": {{
    "python": "class Solution:\\n    def {entry_point}(self, ...):\\n        # Your code here\\n        pass",
    "javascript": "class Solution {{\\n    {entry_point}(...) {{\\n        // Your code here\\n    }}\\n}}",
    "java": "class Solution {{\\n    public ReturnType {entry_point}(...) {{\\n        // Your code here\\n    }}\\n}}",
    "cpp": "class Solution {{\\npublic:\\n    ReturnType {entry_point}(...) {{\\n        // Your code here\\n    }}\\n}};"
  }},
  "driverCode": {{
    "python": "import sys\\nimport json\\n\\n{{USER_CODE}}\\n\\n# Driver code here",
    "javascript": "const fs = require('fs');\\n\\n{{USER_CODE}}\\n\\n// Driver code here",
    "java": "import java.util.*;\\nimport java.io.*;\\n\\n{{USER_CODE}}\\n\\npublic class Main {{\\n    // Driver code here\\n}}",
    "cpp": "#include <iostream>\\n#include <vector>\\n\\n{{USER_CODE}}\\n\\nint main() {{\\n    // Driver code here\\n}}"
  }},
  "testCases": [
    {{
      "input": "test input",
      "expectedOutput": "expected output",
      "isHidden": false
    }},
    {{
      "input": "test input",
      "expectedOutput": "expected output",
      "isHidden": true
    }}
  ]
}}

**Important:**
- Use {{USER_CODE}} placeholder in driverCode where starter code should be inserted
- Generate exactly 5 test cases
- All code must be properly escaped for JSON (use \\n for newlines)
- Ensure the starter code function/class name matches {entry_point} exactly
- Driver code should include ALL code from the full code except the main solution logic
- DO NOT skip any imports, helper functions, parsing logic, or output formatting code
- Starter code should be minimal - only the solution class/function skeleton with placeholder for main logic
- The combination of driver code + starter code should be identical to the original full code"""
        
        return prompt
    
    def _format_test_cases_section(self, test_cases_examples: list) -> str:
        """
        Format test cases examples for the prompt.
        
        Args:
            test_cases_examples: List of test cases from dataset
            
        Returns:
            Formatted string section for prompt
        """
        if not test_cases_examples:
            return ""
        
        section = "\n**Test Cases Examples (Input/Output Format from Dataset):**\n"
        section += "These examples show the EXACT format of input (as string) and expected output (as string) from the LeetCode dataset:\n"
        section += "The input string format is like: 'nums = [3,3], target = 6' (variable assignments as a string)\n"
        section += "The output string format is like: '[0, 1]' or 'None' (string representation of the result)\n\n"
        
        for idx, test_case in enumerate(test_cases_examples[:5], 1):
            if isinstance(test_case, dict):
                input_val = test_case.get('input', '')
                output_val = test_case.get('output', '')
                section += f"Test Case {idx}:\n"
                section += f"  Input (string): {input_val}\n"
                section += f"  Expected Output (string): {output_val}\n\n"
            elif isinstance(test_case, list) and len(test_case) >= 2:
                section += f"Test Case {idx}:\n"
                input_val = test_case[0]
                output_val = test_case[1]
                section += f"  Input (string): {input_val}\n"
                section += f"  Expected Output (string): {output_val}\n\n"
        
        section += "**CRITICAL INPUT/OUTPUT FORMAT REQUIREMENTS:**\n"
        section += "1. Input Format: Test case input should be a STRING in the format shown above (e.g., 'nums = [3,3], target = 6')\n"
        section += "   - This is the exact format that will be provided via stdin in Judge0\n"
        section += "   - The input string contains variable assignments separated by commas\n"
        section += "2. Output Format: Test case expectedOutput should be a STRING matching the exact format shown in examples\n"
        section += "   - Arrays should be formatted as string like '[0, 1]'\n"
        section += "   - None/null should be formatted as string 'None' or 'null'\n"
        section += "   - Numbers should be formatted as string representation\n"
        section += "   - Match the EXACT format from the examples (spacing, brackets, etc.)\n"
        
        return section

