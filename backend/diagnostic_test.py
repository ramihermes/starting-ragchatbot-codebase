"""
Diagnostic script to test the RAG system query flow
This runs against the actual system to identify where 'query failed' errors occur
"""
import sys
import traceback
from config import config
from rag_system import RAGSystem

def test_vector_store_search():
    """Test if vector store search is working"""
    print("="*60)
    print("TEST 1: Vector Store Search")
    print("="*60)
    try:
        from vector_store import VectorStore
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)

        # Test basic search
        results = store.search(query="What is MCP?")

        print(f"[OK] Vector store initialized")
        print(f"  - Documents found: {len(results.documents)}")
        print(f"  - Has error: {results.error}")
        print(f"  - Is empty: {results.is_empty()}")

        if results.error:
            print(f"  [ERROR] ERROR: {results.error}")
            return False

        if results.is_empty():
            print(f"  [WARN] WARNING: No documents in vector store")
        else:
            print(f"  - Sample document: {results.documents[0][:100]}...")
            print(f"  - Sample metadata: {results.metadata[0]}")

        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_search_tool_execute():
    """Test if CourseSearchTool.execute works"""
    print("\n" + "="*60)
    print("TEST 2: CourseSearchTool.execute()")
    print("="*60)
    try:
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        tool = CourseSearchTool(store)

        # Test execution
        result = tool.execute(query="What is MCP?")

        print(f"[OK] CourseSearchTool executed")
        print(f"  - Result type: {type(result)}")
        print(f"  - Result length: {len(result)}")
        print(f"  - Result preview: {result[:200]}...")

        if "error" in result.lower() or "no" in result.lower()[:50]:
            print(f"  [WARN] WARNING: Result might indicate an issue")

        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_generator_without_tools():
    """Test if AIGenerator works for basic queries"""
    print("\n" + "="*60)
    print("TEST 3: AIGenerator without tools")
    print("="*60)
    try:
        from ai_generator import AIGenerator

        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        # Test basic query
        response = generator.generate_response(query="What is 2+2?")

        print(f"[OK] AIGenerator responded")
        print(f"  - Response type: {type(response)}")
        print(f"  - Response: {response}")

        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_generator_with_tools():
    """Test if AIGenerator correctly uses tools"""
    print("\n" + "="*60)
    print("TEST 4: AIGenerator with tool calling")
    print("="*60)
    try:
        from ai_generator import AIGenerator
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import VectorStore

        # Setup components
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(store)
        tool_manager.register_tool(search_tool)

        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        # Test with content question
        print("  Testing content question: 'What is MCP?'")
        response = generator.generate_response(
            query="What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        print(f"[OK] AIGenerator with tools responded")
        print(f"  - Response type: {type(response)}")
        print(f"  - Response length: {len(response)}")
        print(f"  - Response preview: {response[:300]}...")

        # Check if sources were tracked
        sources = tool_manager.get_last_sources()
        print(f"  - Sources tracked: {len(sources)}")
        if sources:
            print(f"  - Sample source: {sources[0]}")

        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_full_rag_system_query():
    """Test the complete RAG system query flow"""
    print("\n" + "="*60)
    print("TEST 5: Full RAG System Query")
    print("="*60)
    try:
        rag = RAGSystem(config)

        # Test content query
        print("  Testing query: 'What is Model Context Protocol?'")
        response, sources = rag.query("What is Model Context Protocol?", session_id="test-123")

        print(f"[OK] RAG system query completed")
        print(f"  - Response type: {type(response)}")
        print(f"  - Response length: {len(response)}")
        print(f"  - Response: {response[:400]}...")
        print(f"  - Sources count: {len(sources)}")
        if sources:
            print(f"  - Sample source: {sources[0]}")

        # Check for failure indicators
        if "query failed" in response.lower():
            print(f"  [ERROR] ERROR: Response contains 'query failed'")
            return False

        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        traceback.print_exc()
        return False

def test_tool_definition_format():
    """Test that tool definitions are correctly formatted"""
    print("\n" + "="*60)
    print("TEST 6: Tool Definition Format")
    print("="*60)
    try:
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import VectorStore

        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        tool = CourseSearchTool(store)
        definition = tool.get_tool_definition()

        print(f"[OK] Tool definition retrieved")
        print(f"  - Name: {definition.get('name')}")
        print(f"  - Has description: {'description' in definition}")
        print(f"  - Has input_schema: {'input_schema' in definition}")

        if 'input_schema' in definition:
            schema = definition['input_schema']
            print(f"  - Schema type: {schema.get('type')}")
            print(f"  - Required fields: {schema.get('required')}")
            print(f"  - Properties: {list(schema.get('properties', {}).keys())}")

        # Check for common issues
        if not definition.get('name'):
            print(f"  [ERROR] ERROR: Tool name is missing")
            return False

        if not definition.get('input_schema'):
            print(f"  [ERROR] ERROR: Input schema is missing")
            return False

        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("RAG SYSTEM DIAGNOSTIC TESTS")
    print("="*60)
    print(f"Config:")
    print(f"  - CHROMA_PATH: {config.CHROMA_PATH}")
    print(f"  - ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
    print(f"  - API Key set: {bool(config.ANTHROPIC_API_KEY)}")
    print()

    results = {
        "Vector Store Search": test_vector_store_search(),
        "CourseSearchTool Execute": test_search_tool_execute(),
        "AIGenerator Basic": test_ai_generator_without_tools(),
        "Tool Definition": test_tool_definition_format(),
        "AIGenerator with Tools": test_ai_generator_with_tools(),
        "Full RAG Query": test_full_rag_system_query(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n[WARN] Some tests failed. Review the output above for details.")
        sys.exit(1)
    else:
        print("\n[OK] All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
