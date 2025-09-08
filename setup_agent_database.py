#!/usr/bin/env python3
"""
Command-line setup script for MongoDB LangGraph Agent
This script can be run with command-line arguments to set up specific components.
"""

import os
import sys
import argparse
import time
from datetime import datetime, timezone
import requests
from typing import List
import PyPDF2

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_mongodb_toolbox_agent import (
    mongo_manager, 
    tool_registry, 
    agent_builder, 
    COLLECTION_NAMES,
    VECTOR_INDEX_CONFIG,
    get_or_create_agent_builder,
)

class DataManager:
    """Manager for data operations (policies, orders, indexes)."""
    
    def __init__(self, mongo_manager):
        self.mongo_manager = mongo_manager
    
    def setup_vector_indexes(self):
        """Setup vector search indexes."""
        import time
        
        required_collections = list(COLLECTION_NAMES.values())
        index_created = False
        
        for collection_name in required_collections:
            print(f"Checking and creating index for collection: {collection_name}")
            collection = self.mongo_manager.get_collection(collection_name)
            
            # Ensure collection exists by inserting and removing a dummy document
            try:
                collection.insert_one({"_dummy": True})
                collection.delete_one({"_dummy": True})
            except Exception:
                # Collection doesn't exist, skip index creation for now
                print(f"Collection {collection_name} does not exist yet. Skipping index creation.")
                continue
            
            try:
                search_indexes = list(collection.list_search_indexes())
                index_exists = any(index.get("name") == VECTOR_INDEX_CONFIG["name"] for index in search_indexes)
            except Exception as e:
                print(f"Could not check search indexes for {collection_name}: {e}")
                index_exists = False
            
            if not index_exists:
                try:
                    # Use the command approach which is more reliable
                    index_definition = {
                        "name": VECTOR_INDEX_CONFIG["name"],
                        "definition": {
                            "mappings": {
                                "dynamic": True,
                                "fields": {
                                    "embedding": {
                                        "dimensions": VECTOR_INDEX_CONFIG["dimensions"],
                                        "similarity": VECTOR_INDEX_CONFIG["similarity"],
                                        "type": "knnVector"
                                    }
                                }
                            }
                        }
                    }
                    
                    # Use the database command to create the search index
                    db = collection.database
                    db.command("createSearchIndexes", collection.name, indexes=[index_definition])
                    print(f"Vector search index created successfully for {collection_name}.")
                    index_created = True
                except Exception as e:
                    print(f"Error creating index for {collection_name}: {e}")
            else:
                print(f"Vector search index already exists for {collection_name}. Skipping creation.")
        
        if index_created:
            print("Pausing for 30 seconds to allow index builds...")
            time.sleep(30)
            print("Resuming after pause.")
        else:
            print("No new indexes were created. Skipping pause.")
    
    def load_policy_documents(self):
        """Load policy documents into MongoDB."""
        policies_collection = self.mongo_manager.get_collection("policies")
        policies_count = policies_collection.count_documents({})
        
        if policies_count > 0:
            print(f"Policies collection is not empty. Skipping document import. Total documents: {policies_count}")
            return
        
        print("Policies collection is empty. Starting document import.")
        
        document_urls = [
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/privacy_policy.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/return_policy.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/shipping_policy.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/terms_of_service.pdf",
            "https://mongodb-llamaindex-demos.s3.us-west-1.amazonaws.com/warranty_policy.pdf"
        ]
        
        temp_dir = "temp_policy_docs"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download and process documents
        for url in document_urls:
            try:
                file_name = os.path.join(temp_dir, url.split("/")[-1])
                response = requests.get(url)
                response.raise_for_status()
                
                with open(file_name, "wb") as f:
                    f.write(response.content)
                
                # Process and store in MongoDB with chunking
                with open(file_name, 'rb') as f:
                    # Extract text from PDF using PyPDF2
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    
                    # Extract text from all pages
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
                    
                    # Normalize PDF extraction artifacts: collapse per-word newlines and extra spaces
                    content = content.replace("\n", " ")
                    content = ' '.join(content.split())
                    
                    if not content:
                        print(f"Warning: No text extracted from {file_name}")
                        continue
                    
                    print(f"Processing {file_name}: {len(content)} characters")
                    
                    # Pre-chunk to ~5000 characters (no overlap) to match SDK expectations
                    def chunk_text_by_chars(text: str, max_len: int = 500) -> List[str]:
                        chunks = []
                        start = 0
                        text_len = len(text)
                        while start < text_len:
                            end = min(start + max_len, text_len)
                            # Try not to cut in the middle of a word
                            if end < text_len:
                                last_space = text.rfind(" ", start, end)
                                if last_space != -1 and last_space > start + int(max_len * 0.6):
                                    end = last_space
                            chunks.append(text[start:end])
                            start = end
                        return chunks

                    chunks = chunk_text_by_chars(content, max_len=500)

                    # Use Voyage contextualized chunk embeddings (voyage-context-3)
                    result = self.mongo_manager.voyage_client.contextualized_embed(
                        inputs=[chunks],
                        input_type="document",
                        model="voyage-context-3",
                        output_dimension=VECTOR_INDEX_CONFIG["dimensions"]
                    )

                    # SDK returns ContextualizedEmbeddingsObject with .results list
                    res0 = result.results[0] if getattr(result, "results", None) else None
                    if not res0:
                        print(f"Warning: No embeddings returned for {file_name}")
                        continue

                    chunk_embeddings = res0.embeddings
                    chunk_texts = res0.chunk_texts or chunks

                    inserted = 0
                    for idx, (chunk_text, chunk_vector) in enumerate(zip(chunk_texts, chunk_embeddings)):
                        if not chunk_text:
                            continue

                        doc = {
                            "text": chunk_text,
                            "metadata": {
                                "file_name": os.path.basename(file_name),
                                "document_type": "policy",
                                "original_length": len(content),
                                "chunk_index": idx,
                            },
                            "embedding": chunk_vector
                        }
                        policies_collection.insert_one(doc)
                        inserted += 1

                    print(f"Stored {inserted} chunks from {file_name} in MongoDB using voyage-context-3")
                    
            except Exception as e:
                print(f"Error processing {url}: {e}")
        
        policies_count = policies_collection.count_documents({})
        print(f"Successfully ingested all PDF documents. Total documents: {policies_count}")
    

    
    def create_dummy_order_data(self):
        """Create dummy order data for testing."""
        orders_collection = self.mongo_manager.get_collection("orders")
        
        if orders_collection.count_documents({}) > 0:
            print("Orders collection is not empty. Skipping insertion of fake orders.")
            return
        
        fake_orders = [
            {
                "order_id": 101,
                "order_date": datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc),
                "status": "Shipped",
                "total_amount": 1315.75,
                "shipping_address": "123 Main St, Anytown, CA 91234",
                "payment_method": "Credit Card",
                "items": [
                    {"name": "Laptop", "price": 1290.00},
                    {"name": "Mouse", "price": 25.75},
                ],
            },
            {
                "order_id": 102,
                "order_date": datetime(2023, 10, 25, 14, 30, 0, tzinfo=timezone.utc),
                "status": "Processing",
                "total_amount": 55.00,
                "shipping_address": "456 Oak Ave, Somewhere, NY 54321",
                "payment_method": "PayPal",
                "items": [
                    {"name": "Keyboard", "price": 75.00},
                ],
            },
            {
                "order_id": 103,
                "order_date": datetime(2023, 10, 25, 14, 30, 0, tzinfo=timezone.utc),
                "status": "Processing",
                "total_amount": 35.00,
                "shipping_address": "789 Pine Rd, Elsewhere, TX 67890",
                "payment_method": "Debit Card",
                "items": [
                    {"name": "Monitor", "price": 250.00},
                ],
            },
        ]
        
        orders_collection.insert_many(fake_orders)
        print(f"Inserted {len(fake_orders)} fake orders.")

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_mongodb_toolbox_agent import (
    mongo_manager, 
    tool_registry, 
    agent_builder, 
    COLLECTION_NAMES,
    VECTOR_INDEX_CONFIG
)

class CLISetupManager:
    """Command-line setup manager for the MongoDB agent system."""
    
    def __init__(self):
        self.mongo_manager = mongo_manager
        self.tool_registry = tool_registry
        # Agent builder not required for setup steps; can be initialized by main app
        self.agent_builder = None
        self.data_manager = DataManager(mongo_manager)
    
    def check_mongodb_connection(self) -> bool:
        """Check MongoDB connection."""
        print("Checking MongoDB connection...")
        try:
            # Use the database from the config
            db = self.mongo_manager.config.db
            self.mongo_manager.client.admin.command('ping')
            print(f"✅ MongoDB connection successful! Database: {db.name}")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    def setup_tools(self):
        """Set up tools in MongoDB."""
        print("Setting up tools...")
        try:
            from langgraph_mongodb_toolbox_agent import register_all_tools
            register_all_tools()
            print("✅ Tools setup completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error setting up tools: {e}")
            return False
    
    def setup_vector_indexes(self):
        """Set up vector search indexes."""
        print("Setting up vector indexes...")
        try:
            self.data_manager.setup_vector_indexes()
            print("✅ Vector indexes setup completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error setting up vector indexes: {e}")
            return False
    
    def setup_policy_documents(self):
        """Set up policy documents."""
        print("Setting up policy documents...")
        try:
            self.data_manager.load_policy_documents()
            print("✅ Policy documents setup completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error setting up policy documents: {e}")
            return False
    
    def setup_dummy_data(self):
        """Set up dummy order data."""
        print("Setting up dummy data...")
        try:
            self.data_manager.create_dummy_order_data()
            print("✅ Dummy data setup completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error setting up dummy data: {e}")
            return False
    
    def show_collection_status(self):
        """Show the status of all collections."""
        print("Checking collection status...")
        try:
            for collection_name in COLLECTION_NAMES.values():
                collection = self.mongo_manager.get_collection(collection_name)
                count = collection.count_documents({})
                print(f"📊 {collection_name}: {count} documents")
            print("✅ Collection status check completed!")
            return True
        except Exception as e:
            print(f"❌ Error checking collection status: {e}")
            return False
    
    def run_full_setup(self):
        """Run the complete setup process."""
        print("🚀 Running full setup...")
        
        # Check MongoDB connection first
        if not self.check_mongodb_connection():
            print("❌ Cannot proceed without MongoDB connection")
            return False
        
        # Run all setup steps
        steps = [
            ("Tools", self.setup_tools),
            ("Vector Indexes", self.setup_vector_indexes),
            ("Policy Documents", self.setup_policy_documents),
            ("Dummy Data", self.setup_dummy_data),
            ("Collection Status", self.show_collection_status),
        ]
        
        results = []
        for step_name, step_func in steps:
            print(f"\n--- {step_name} ---")
            result = step_func()
            results.append((step_name, result))
        
        # Summary
        print("\n" + "="*50)
        print("SETUP SUMMARY")
        print("="*50)
        for step_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{step_name}: {status}")
        
        all_passed = all(result for _, result in results)
        if all_passed:
            print("\n🎉 All setup steps completed successfully!")
        else:
            print("\n⚠️  Some setup steps failed. Check the output above.")
        
        return all_passed

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup script for MongoDB LangGraph Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_agent_database.py --full                    # Run full setup
  python setup_agent_database.py --tools --indexes         # Setup tools and indexes only
  python setup_agent_database.py --check-connection        # Check MongoDB connection only
        """
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run complete setup (all components)"
    )
    parser.add_argument(
        "--check-connection", 
        action="store_true", 
        help="Check MongoDB connection"
    )
    parser.add_argument(
        "--tools", 
        action="store_true", 
        help="Setup tools"
    )
    parser.add_argument(
        "--indexes", 
        action="store_true", 
        help="Setup vector indexes"
    )
    parser.add_argument(
        "--policies", 
        action="store_true", 
        help="Setup policy documents"
    )
    parser.add_argument(
        "--dummy-data", 
        action="store_true", 
        help="Setup dummy data"
    )
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Show collection status"
    )
    # Removed agent test option from setup script
    
    args = parser.parse_args()
    
    # Check if required environment variables are set
    required_env_vars = ["MONGODB_URI", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables before running the setup.")
        sys.exit(1)
    
    # Create setup manager
    setup_manager = CLISetupManager()
    
    # Determine what to run
    if args.full:
        success = setup_manager.run_full_setup()
        sys.exit(0 if success else 1)
    
    # Individual components
    if args.check_connection:
        success = setup_manager.check_mongodb_connection()
        sys.exit(0 if success else 1)
    
    if args.tools:
        success = setup_manager.setup_tools()
        sys.exit(0 if success else 1)
    
    if args.indexes:
        success = setup_manager.setup_vector_indexes()
        sys.exit(0 if success else 1)
    
    if args.policies:
        success = setup_manager.setup_policy_documents()
        sys.exit(0 if success else 1)
    
    if args.dummy_data:
        success = setup_manager.setup_dummy_data()
        sys.exit(0 if success else 1)
    
    if args.status:
        success = setup_manager.show_collection_status()
        sys.exit(0 if success else 1)
    
    # Agent test removed
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
