import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from dotenv import load_dotenv
from schemas import CodingQuestion

# Load environment variables
load_dotenv()


class MongoDBHandler:
    """
    MongoDB handler class for storing and managing generated coding questions.
    Prevents duplicate questions by checking existing entries before storing new ones.
    """
    
    def __init__(self, mongo_db_url: Optional[str] = None, db_name: str = "algocode_generator", collection_name: str = "generated_questions"):
        """
        Initialize MongoDB connection.
        
        Args:
            mongo_db_url: MongoDB connection URL. If None, will try to get from MONGO_DB_URL environment variable
            db_name: Name of the database to use (default: "algocode_generator")
            collection_name: Name of the collection to use (default: "generated_questions")
        """
        # Get MongoDB URL from environment or parameter
        self.mongo_db_url = mongo_db_url or os.getenv("MONGO_DB_URL")
        if not self.mongo_db_url:
            raise ValueError("MongoDB URL is required. Provide it as argument or set MONGO_DB_URL environment variable.")
        
        # Initialize MongoDB client
        try:
            self.client = MongoClient(self.mongo_db_url)
            # Test connection
            self.client.admin.command('ping')
            print("✓ Successfully connected to MongoDB")
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
        
        # Get database and collection
        self.db: Database = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        
        # Create indexes for efficient querying
        self._create_indexes()
        
        print(f"✓ Using database: {db_name}, collection: {collection_name}")
    
    def _create_indexes(self):
        """Create indexes for efficient querying and to prevent duplicates."""
        # Get existing indexes to check what needs to be dropped
        existing_indexes = list(self.collection.list_indexes())
        index_names = [idx["name"] for idx in existing_indexes]
        
        # Drop old unique index on title if it exists (from previous version)
        # Check for both "title_1" and any index with title as key
        for idx in existing_indexes:
            if "title" in idx.get("key", {}) and idx.get("unique", False):
                try:
                    self.collection.drop_index(idx["name"])
                    print(f"✓ Dropped old unique index on title: {idx['name']}")
                except Exception:
                    pass
        
        # Create unique index on question_id to prevent duplicates
        # Drop existing question_id index if it exists and recreate
        if "question_id_1" in index_names:
            try:
                self.collection.drop_index("question_id_1")
            except Exception:
                pass
        
        try:
            self.collection.create_index("question_id", unique=True, name="question_id_unique")
        except Exception as e:
            # If it already exists with different name, that's okay
            if "already exists" not in str(e).lower():
                print(f"⚠ Note: question_id index: {str(e)}")
        
        # Create index on title for faster lookups (non-unique, as titles might repeat)
        # Drop existing title index if it's unique
        if "title_1" in index_names:
            try:
                # Check if it's unique before dropping
                for idx in existing_indexes:
                    if idx["name"] == "title_1" and idx.get("unique", False):
                        self.collection.drop_index("title_1")
                        break
            except Exception:
                pass
        
        try:
            self.collection.create_index("title", name="title_index")
        except Exception as e:
            if "already exists" not in str(e).lower() and "duplicate key" not in str(e).lower():
                print(f"⚠ Note: title index: {str(e)}")
        
        # Create other indexes (non-unique, safe to create multiple times)
        for field in ["functionName", "difficulty", "createdAt"]:
            try:
                self.collection.create_index(field)
            except Exception:
                # Index already exists, that's fine
                pass
        
        print("✓ Created/verified database indexes")
    
    def question_exists(self, question_id: str) -> bool:
        """
        Check if a question with the given question_id already exists in the database.
        
        Args:
            question_id: The question_id from the dataset to check
            
        Returns:
            True if question exists, False otherwise
        """
        existing = self.collection.find_one({"question_id": question_id})
        return existing is not None
    
    def get_existing_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an existing question by question_id.
        
        Args:
            question_id: The question_id from the dataset to retrieve
            
        Returns:
            Dictionary containing the question data if found, None otherwise
        """
        question = self.collection.find_one({"question_id": question_id})
        if question:
            # Convert ObjectId to string for JSON serialization
            question["_id"] = str(question["_id"])
        return question
    
    def get_all_existing_question_ids(self) -> List[str]:
        """
        Get all existing question_ids from the database.
        
        Returns:
            List of all question_ids
        """
        question_ids = self.collection.find({}, {"question_id": 1, "_id": 0})
        return [doc["question_id"] for doc in question_ids if doc.get("question_id")]
    
    def store_question(self, question_data: Dict[str, Any], question_id: str) -> Optional[str]:
        """
        Store a new question in the database.
        Will skip if a question with the same question_id already exists.
        
        Args:
            question_data: Dictionary containing question data (should match CodingQuestion schema)
            question_id: The question_id from the dataset (used as unique identifier)
            
        Returns:
            MongoDB document ID if successfully stored, None if duplicate or error
        """
        # Check if question_id is provided
        if not question_id:
            print("✗ Question ID is required")
            return None
        
        # Check if question already exists
        if self.question_exists(question_id):
            print(f"⚠ Question with ID '{question_id}' already exists. Skipping...")
            return None
        
        # Validate question data against schema
        try:
            validated_question = CodingQuestion(**question_data)
        except Exception as e:
            print(f"✗ Schema validation failed: {str(e)}")
            return None
        
        # Prepare document for insertion
        document = validated_question.model_dump(exclude_none=True)
        document["question_id"] = question_id  # Store the dataset question_id
        document["createdAt"] = datetime.utcnow()
        document["updatedAt"] = datetime.utcnow()
        
        # Remove _id if present (MongoDB will generate it)
        if "_id" in document:
            del document["_id"]
        
        title = question_data.get("title", "Unknown")
        try:
            result = self.collection.insert_one(document)
            print(f"✓ Successfully stored question: '{title}' (Question ID: {question_id}, MongoDB ID: {result.inserted_id})")
            return str(result.inserted_id)
        except DuplicateKeyError:
            print(f"⚠ Question with ID '{question_id}' already exists (duplicate key error). Skipping...")
            return None
        except Exception as e:
            print(f"✗ Error storing question '{title}' (ID: {question_id}): {str(e)}")
            return None
    
    def store_questions_batch(self, questions: List[Dict[str, Any]], question_ids: List[str]) -> Dict[str, Any]:
        """
        Store multiple questions in batch.
        
        Args:
            questions: List of question dictionaries
            question_ids: List of question_ids corresponding to each question
            
        Returns:
            Dictionary with statistics: stored, skipped, failed
        """
        if len(questions) != len(question_ids):
            raise ValueError("Number of questions must match number of question_ids")
        
        stats = {
            "stored": 0,
            "skipped": 0,
            "failed": 0,
            "stored_ids": []
        }
        
        for question, q_id in zip(questions, question_ids):
            result_id = self.store_question(question, q_id)
            if result_id:
                stats["stored"] += 1
                stats["stored_ids"].append(result_id)
            elif self.question_exists(q_id):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
        
        return stats
    
    def filter_existing_questions(self, question_ids: List[str]) -> List[str]:
        """
        Filter out question_ids that already exist in the database.
        
        Args:
            question_ids: List of question_ids to filter
            
        Returns:
            List of question_ids that don't exist in the database
        """
        existing_ids = set(self.get_all_existing_question_ids())
        new_question_ids = [
            q_id for q_id in question_ids 
            if q_id and q_id not in existing_ids
        ]
        return new_question_ids
    
    def get_question_count(self) -> int:
        """
        Get the total number of questions in the database.
        
        Returns:
            Total count of questions
        """
        return self.collection.count_documents({})
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Get all questions of a specific difficulty level.
        
        Args:
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            List of question dictionaries
        """
        questions = list(self.collection.find({"difficulty": difficulty.lower()}))
        # Convert ObjectId to string
        for q in questions:
            q["_id"] = str(q["_id"])
        return questions
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a question by its question_id from the dataset.
        
        Args:
            question_id: The question_id from the dataset
            
        Returns:
            Dictionary containing the question data if found, None otherwise
        """
        question = self.collection.find_one({"question_id": question_id})
        if question:
            # Convert ObjectId to string
            question["_id"] = str(question["_id"])
        return question
    
    def close_connection(self):
        """Close the MongoDB connection."""
        self.client.close()
        print("✓ MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()

