"""
Example usage of MongoDB integration with LeetCodeQuestionGenerator.

Make sure to set MONGO_DB_URL in your .env file:
MONGO_DB_URL=mongodb://localhost:27017/
# or
MONGO_DB_URL=mongodb+srv://username:password@cluster.mongodb.net/
"""

from main import LeetCodeQuestionGenerator
from mongodb_handler import MongoDBHandler

# Example 1: Using LeetCodeQuestionGenerator with MongoDB (automatic duplicate checking)
def example_with_generator():
    """Example of using the generator with MongoDB integration."""
    # Initialize generator (MongoDB is enabled by default)
    generator = LeetCodeQuestionGenerator()
    
    # Load dataset
    generator.load_dataset()
    
    # Process questions - MongoDB will automatically check for duplicates
    # and store new questions
    results = generator.process_questions(easy=2, medium=1, hard=0)
    
    # Results will include MongoDB status for each question
    for result in results:
        if result.get("success"):
            print(f"MongoDB status: {result.get('mongodb_status', 'N/A')}")
    
    # Close MongoDB connection
    if generator.mongodb_handler:
        generator.mongodb_handler.close_connection()


# Example 2: Using MongoDBHandler directly
def example_direct_mongodb():
    """Example of using MongoDBHandler directly."""
    # Initialize MongoDB handler
    db_handler = MongoDBHandler()
    
    # Check if a question exists by question_id
    sample_question_id = "1"  # Example question_id from dataset
    if db_handler.question_exists(sample_question_id):
        print(f"Question with ID '{sample_question_id}' already exists!")
    else:
        print(f"Question with ID '{sample_question_id}' does not exist.")
    
    # Get all existing question IDs
    existing_ids = db_handler.get_all_existing_question_ids()
    print(f"Total existing questions: {len(existing_ids)}")
    
    # Get a question by question_id
    question = db_handler.get_question_by_id(sample_question_id)
    if question:
        print(f"Found question: {question.get('title', 'N/A')}")
    
    # Get questions by difficulty
    easy_questions = db_handler.get_questions_by_difficulty("easy")
    print(f"Easy questions in database: {len(easy_questions)}")
    
    # Store a new question
    sample_question = {
        "title": "Sample Question",
        "description": "This is a sample question",
        "difficulty": "easy",
        "tags": ["array", "hash-table"],
        "functionName": "sampleFunction",
        "functionSignature": "def sampleFunction(nums: List[int]) -> int:",
        "examples": [
            {
                "input": "[1, 2, 3]",
                "output": "6",
                "explanation": "Sum of all numbers"
            }
        ],
        "constraints": ["1 <= nums.length <= 100"],
        "starterCode": {
            "python": "def sampleFunction(nums):\n    pass",
            "javascript": "function sampleFunction(nums) {\n    \n}",
            "java": "class Solution {\n    public int sampleFunction(int[] nums) {\n        \n    }\n}",
            "cpp": "class Solution {\npublic:\n    int sampleFunction(vector<int>& nums) {\n        \n    }\n};"
        },
        "driverCode": {
            "python": "import sys\nimport json\n\n{{USER_CODE}}\n\nif __name__ == '__main__':\n    nums = json.loads(sys.stdin.read())\n    print(sampleFunction(nums))",
            "javascript": "const readline = require('readline');\n\n{{USER_CODE}}\n\nconst rl = readline.createInterface({\n    input: process.stdin,\n    output: process.stdout\n});\n\nrl.on('line', (input) => {\n    const nums = JSON.parse(input);\n    console.log(sampleFunction(nums));\n    rl.close();\n});",
            "java": "import java.util.*;\nimport java.io.*;\n\n{{USER_CODE}}\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);\n        String input = scanner.nextLine();\n        int[] nums = Arrays.stream(input.replaceAll(\"[\\\\[\\\\]]\", \"\").split(\",\"))\n                          .mapToInt(Integer::parseInt)\n                          .toArray();\n        Solution solution = new Solution();\n        System.out.println(solution.sampleFunction(nums));\n    }\n}",
            "cpp": "#include <iostream>\n#include <vector>\n#include <sstream>\n#include <string>\n\n{{USER_CODE}}\n\nint main() {\n    std::string input;\n    std::getline(std::cin, input);\n    std::vector<int> nums;\n    std::istringstream iss(input);\n    int num;\n    while (iss >> num) {\n        nums.push_back(num);\n    }\n    Solution solution;\n    std::cout << solution.sampleFunction(nums) << std::endl;\n    return 0;\n}"
        },
        "testCases": [
            {
                "input": "[1, 2, 3]",
                "expectedOutput": "6",
                "isHidden": False
            },
            {
                "input": "[4, 5, 6]",
                "expectedOutput": "15",
                "isHidden": True
            }
        ]
    }
    
    # Store a new question with a question_id
    sample_question_id = "999"  # Example question_id from dataset
    stored_id = db_handler.store_question(sample_question, sample_question_id)
    if stored_id:
        print(f"Question stored with MongoDB ID: {stored_id}")
    else:
        print("Failed to store question (may be duplicate or invalid)")
    
    # Close connection
    db_handler.close_connection()


# Example 3: Filter existing questions before processing
def example_filter_existing():
    """Example of filtering out existing questions before generation."""
    generator = LeetCodeQuestionGenerator()
    generator.load_dataset()
    
    # Get existing question IDs from MongoDB
    if generator.mongodb_handler:
        existing_ids = set(generator.mongodb_handler.get_all_existing_question_ids())
        print(f"Found {len(existing_ids)} existing questions in database")
        
        # You can filter question IDs before processing to avoid generating
        # questions that already exist
        # For example, if you have a list of question_ids to process:
        # question_ids_to_process = ["1", "2", "3", "4", "5"]
        # new_question_ids = generator.mongodb_handler.filter_existing_questions(question_ids_to_process)
        # print(f"New question IDs to process: {new_question_ids}")
    
    # Process questions - duplicates will be automatically skipped by question_id
    results = generator.process_questions(easy=5, medium=3, hard=1)
    
    if generator.mongodb_handler:
        generator.mongodb_handler.close_connection()


if __name__ == "__main__":
    print("Example 1: Using generator with MongoDB")
    # example_with_generator()
    
    print("\nExample 2: Using MongoDBHandler directly")
    # example_direct_mongodb()
    
    print("\nExample 3: Filter existing questions")
    # example_filter_existing()
    
    print("\nUncomment the examples above to run them.")

