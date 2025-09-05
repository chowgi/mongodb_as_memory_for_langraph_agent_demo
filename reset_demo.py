#!/usr/bin/env python3
"""Reset demo by clearing conversation memory from LangGraph checkpointing collections."""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

def get_mongodb_client():
    """Get MongoDB client connection."""
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    
    return MongoClient(mongodb_uri)

def show_collection_contents(collection, collection_name):
    """Display contents of a collection."""
    print(f"\n📋 Contents of '{collection_name}' collection:")
    print("-" * 50)
    
    documents = list(collection.find({}))
    
    if not documents:
        print("   (No documents found)")
        return 0
    
    print(f"   Found {len(documents)} document(s):")
    for i, doc in enumerate(documents, 1):
        # Safely convert ObjectId to string
        doc_id = str(doc.get('_id', 'N/A'))
        print(f"   {i}. Document ID: {doc_id}")
        
        # Show key fields for checkpoints
        if 'thread_id' in doc:
            thread_id = str(doc['thread_id']) if doc['thread_id'] else 'N/A'
            print(f"      Thread ID: {thread_id}")
        if 'checkpoint' in doc:
            checkpoint = doc['checkpoint']
            if isinstance(checkpoint, dict) and 'messages' in checkpoint:
                messages = checkpoint['messages']
                if isinstance(messages, list):
                    print(f"      Messages: {len(messages)} message(s)")
                else:
                    print(f"      Messages: {type(messages).__name__}")
        if 'timestamp' in doc:
            timestamp = str(doc['timestamp']) if doc['timestamp'] else 'N/A'
            print(f"      Timestamp: {timestamp}")
        
        print()
    
    return len(documents)

def reset_conversation_memory(show_contents=True, interactive=False):
    """
    Reset conversation memory by clearing all documents from checkpointing collections and demo data.
    
    Args:
        show_contents (bool): Whether to show collection contents before deletion
        interactive (bool): Whether to prompt for confirmation
    
    Returns:
        int: Number of documents deleted
    """
    
    try:
        # Connect to MongoDB
        client = get_mongodb_client()
        
        # Access the databases
        checkpointing_db = client['checkpointing_db']
        demo_db = client['langgraph_agent_demo']
        
        # Collections to clear
        checkpointing_collections = ['checkpoints', 'checkpoint_writes']
        demo_collections = ['returns']  # Add other demo collections here if needed
        
        total_deleted = 0
        
        # Clear checkpointing collections
        for collection_name in checkpointing_collections:
            try:
                collection = checkpointing_db[collection_name]
                doc_count = collection.count_documents({})
                
                if doc_count > 0:
                    # Delete all documents
                    result = collection.delete_many({})
                    total_deleted += result.deleted_count
            except Exception as e:
                continue
        
        # Clear demo collections
        for collection_name in demo_collections:
            try:
                collection = demo_db[collection_name]
                doc_count = collection.count_documents({})
                
                if doc_count > 0:
                    # Delete all documents
                    result = collection.delete_many({})
                    total_deleted += result.deleted_count
            except Exception as e:
                continue
        
        return total_deleted
        
    except Exception as e:
        print(f"❌ Error clearing collections: {e}")
        return 0
    finally:
        if 'client' in locals():
            client.close()

def clear_checkpointing_collections():
    """Clear all documents from LangGraph checkpointing collections."""
    return reset_conversation_memory(show_contents=True, interactive=True)

def show_checkpointing_stats():
    """Show statistics about the collections that will be reset."""
    
    try:
        client = get_mongodb_client()
        checkpointing_db = client['checkpointing_db']
        demo_db = client['langgraph_agent_demo']
        
        total_docs = 0
        
        # Collections that will be reset
        checkpointing_collections = ['checkpoints', 'checkpoint_writes']
        demo_collections = ['returns']
        
        # Checkpointing database stats (only relevant collections)
        print("📊 Collections to be reset:")
        print("=" * 40)
        
        for collection_name in checkpointing_collections:
            try:
                collection = checkpointing_db[collection_name]
                doc_count = collection.count_documents({})
                total_docs += doc_count
                print(f"  - {collection_name}: {doc_count} document(s)")
            except Exception as e:
                print(f"  - {collection_name}: Error accessing collection ({e})")
        
        # Demo database stats (only relevant collections)
        for collection_name in demo_collections:
            try:
                collection = demo_db[collection_name]
                doc_count = collection.count_documents({})
                total_docs += doc_count
                print(f"  - {collection_name}: {doc_count} document(s)")
            except Exception as e:
                print(f"  - {collection_name}: Error accessing collection ({e})")
        
        print(f"\nTotal documents to be deleted: {total_docs}")
        
        return total_docs
        
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
        return 0
    finally:
        if 'client' in locals():
            client.close()

def main():
    """Main function to reset demo conversation memory and data."""
    
    print("🤖 LangGraph Demo Reset Tool")
    print("=" * 50)
    
    # Show collections to be reset
    print("\n📈 Collections to be reset:")
    total_docs = show_checkpointing_stats()
    
    if total_docs == 0:
        print("\nℹ️  No data found to reset.")
        return
    
    # Ask for confirmation
    print("\n⚠️  This will delete conversation memory and demo data!")
    print("   This includes:")
    print("   - All saved conversation threads")
    print("   - All checkpoint data")
    print("   - All return requests")
    
    response = input("\n❓ Are you sure you want to continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        print("\n🔄 Proceeding with reset...")
        deleted_count = reset_conversation_memory(show_contents=False, interactive=False)
        
        if deleted_count > 0:
            print(f"✅ Successfully deleted {deleted_count} document(s)")
        else:
            print("ℹ️  No data was deleted.")
            
    else:
        print("❌ Reset cancelled.")

if __name__ == "__main__":
    main()
