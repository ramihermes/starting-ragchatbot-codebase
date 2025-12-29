"""
Integration tests for RAG system query flow
Tests end-to-end functionality with real components where possible
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import RAGSystem
from config import Config


class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY = "test-api-key"
    ANTHROPIC_MODEL = "claude-3-sonnet"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = None  # Will be set to temp dir in tests


class TestRAGSystemQuery:
    """Test RAG system query handling"""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_chroma_dir):
        """Create mock config with temp directory"""
        config = MockConfig()
        config.CHROMA_PATH = temp_chroma_dir
        return config

    def test_query_with_content_question_calls_tool(self, mock_config):
        """Test that content-related questions trigger tool usage"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            # Setup mock AI generator
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            # Mock response simulating tool was used
            mock_ai_gen.generate_response.return_value = "MCP is the Model Context Protocol used for..."

            # Create RAG system
            rag = RAGSystem(mock_config)

            # Simulate that search tool found content
            rag.search_tool.last_sources = [
                {"text": "MCP Course - Lesson 1", "url": "https://example.com/lesson1"}
            ]

            # Execute query
            response, sources = rag.query("What is MCP?", session_id="test-session")

            # Verify AI generator was called with tools
            mock_ai_gen.generate_response.assert_called_once()
            call_kwargs = mock_ai_gen.generate_response.call_args[1]

            assert "tools" in call_kwargs
            assert call_kwargs["tool_manager"] is not None
            assert len(call_kwargs["tools"]) > 0

            # Verify response returned
            assert "MCP" in response

    def test_query_with_general_question_may_skip_tool(self, mock_config):
        """Test that general knowledge questions may not need tool usage"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            # Mock response for general question
            mock_ai_gen.generate_response.return_value = "The sky appears blue due to Rayleigh scattering."

            rag = RAGSystem(mock_config)

            # No sources set (tool not used)
            rag.search_tool.last_sources = []

            response, sources = rag.query("Why is the sky blue?", session_id="test-session")

            # Should still call AI generator
            mock_ai_gen.generate_response.assert_called_once()

            # Sources should be empty
            assert len(sources) == 0

    def test_query_returns_sources_from_tool(self, mock_config):
        """Test that sources from tool search are properly returned"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen
            mock_ai_gen.generate_response.return_value = "Here's information about the course."

            rag = RAGSystem(mock_config)

            # Set up sources that would come from tool execution
            expected_sources = [
                {"text": "Course A - Lesson 1", "url": "https://example.com/a/1"},
                {"text": "Course B - Lesson 2", "url": "https://example.com/b/2"}
            ]
            rag.search_tool.last_sources = expected_sources

            response, sources = rag.query("Tell me about the course", session_id="test-session")

            # Verify sources are returned
            assert sources == expected_sources

    def test_query_session_management(self, mock_config):
        """Test that conversation history is tracked across queries"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen
            mock_ai_gen.generate_response.return_value = "Response"

            rag = RAGSystem(mock_config)
            session_id = "test-session-123"

            # First query
            rag.query("First question", session_id=session_id)

            # Second query - should have history
            rag.query("Follow up question", session_id=session_id)

            # Verify second call included history
            assert mock_ai_gen.generate_response.call_count == 2
            second_call_kwargs = mock_ai_gen.generate_response.call_args_list[1][1]

            # History should be included
            assert second_call_kwargs["conversation_history"] is not None
            history = second_call_kwargs["conversation_history"]
            assert "First question" in history

    def test_query_without_session_id(self, mock_config):
        """Test query without providing session ID"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen
            mock_ai_gen.generate_response.return_value = "Response"

            rag = RAGSystem(mock_config)

            # Query without session
            response, sources = rag.query("What is AI?", session_id=None)

            # Should still work
            assert response == "Response"
            mock_ai_gen.generate_response.assert_called_once()

            # But no history should be passed
            call_kwargs = mock_ai_gen.generate_response.call_args[1]
            assert call_kwargs["conversation_history"] is None

    def test_sources_reset_after_retrieval(self, mock_config):
        """Test that sources are reset after being retrieved"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen
            mock_ai_gen.generate_response.return_value = "Response"

            rag = RAGSystem(mock_config)

            # Set sources for first query
            rag.search_tool.last_sources = [
                {"text": "Source 1", "url": "https://example.com/1"}
            ]

            response1, sources1 = rag.query("First query")
            assert len(sources1) == 1

            # Second query without setting new sources
            # Sources should be empty (reset happened)
            response2, sources2 = rag.query("Second query")
            assert len(sources2) == 0

    def test_tool_manager_has_search_tool_registered(self, mock_config):
        """Test that CourseSearchTool is properly registered with ToolManager"""
        rag = RAGSystem(mock_config)

        # Verify tool is registered
        tool_definitions = rag.tool_manager.get_tool_definitions()
        assert len(tool_definitions) >= 1

        # Verify it's the search tool
        search_tool_def = next(
            (t for t in tool_definitions if t["name"] == "search_course_content"),
            None
        )
        assert search_tool_def is not None
        assert "description" in search_tool_def
        assert "input_schema" in search_tool_def


class TestRAGSystemWithRealVectorStore:
    """Integration tests using real VectorStore (but mocked AI)"""

    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_chroma_dir):
        """Create mock config with temp directory"""
        config = MockConfig()
        config.CHROMA_PATH = temp_chroma_dir
        return config

    def test_query_with_empty_vector_store(self, mock_config):
        """Test querying when vector store has no data"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('ai_generator.anthropic.Anthropic') as mock_anthropic:

            # Setup mocks
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            # Mock the tool call flow
            from test_ai_generator import MockContentBlock, MockResponse

            # Tool use response
            tool_response = MockResponse(
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "nonexistent"},
                        block_id="tool_1"
                    )
                ],
                stop_reason="tool_use"
            )

            # Final response after tool returns no results
            final_response = MockResponse(
                content=[MockContentBlock("text", text="I don't have information about that topic.")],
                stop_reason="end_turn"
            )

            mock_ai_gen.generate_response.side_effect = [
                # Simulate the _handle_tool_execution flow
                "I don't have information about that topic."
            ]

            rag = RAGSystem(mock_config)

            # Query empty store
            response, sources = rag.query("What is quantum computing?")

            # Should get a response even with no data
            assert len(response) > 0
            # Sources should be empty since vector store has no data
            assert len(sources) == 0

    def test_error_handling_in_query(self, mock_config):
        """Test error handling when something goes wrong"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            # Simulate AI generator error
            mock_ai_gen.generate_response.side_effect = Exception("API Error")

            rag = RAGSystem(mock_config)

            # Query should raise the exception
            with pytest.raises(Exception) as exc_info:
                rag.query("Test query")

            assert "API Error" in str(exc_info.value)


class TestRAGSystemPromptConstruction:
    """Test how RAG system constructs prompts"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = MockConfig()
        config.CHROMA_PATH = tempfile.mkdtemp()
        return config

    def test_query_prompt_format(self, mock_config):
        """Test that query is properly formatted in prompt"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen
            mock_ai_gen.generate_response.return_value = "Response"

            rag = RAGSystem(mock_config)
            user_query = "What is machine learning?"

            rag.query(user_query)

            # Check the prompt passed to AI
            call_args = mock_ai_gen.generate_response.call_args[1]
            prompt = call_args["query"]

            assert "course materials" in prompt.lower()
            assert user_query in prompt

    def test_conversation_context_passed(self, mock_config):
        """Test that conversation context is properly passed"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen
            mock_ai_gen.generate_response.return_value = "Response"

            rag = RAGSystem(mock_config)
            session_id = "test-session"

            # First exchange
            rag.query("What is AI?", session_id=session_id)

            # Second exchange
            rag.query("Tell me more", session_id=session_id)

            # Verify history is passed
            second_call = mock_ai_gen.generate_response.call_args_list[1]
            history = second_call[1]["conversation_history"]

            assert history is not None
            assert len(history) > 0


class TestRAGSystemRealScenarios:
    """Test realistic user scenarios"""

    @pytest.fixture
    def mock_config(self):
        config = MockConfig()
        config.CHROMA_PATH = tempfile.mkdtemp()
        return config

    def test_content_question_workflow(self, mock_config):
        """Test complete workflow for a content question"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            # Simulate realistic response
            mock_ai_gen.generate_response.return_value = (
                "Model Context Protocol (MCP) is a standardized way for AI systems "
                "to interact with external data sources and tools."
            )

            rag = RAGSystem(mock_config)

            # Set realistic sources
            rag.search_tool.last_sources = [
                {
                    "text": "Introduction to MCP - Lesson 1",
                    "url": "https://learn.deeplearning.ai/mcp/lesson/1"
                }
            ]

            response, sources = rag.query(
                "What is Model Context Protocol?",
                session_id="user-123"
            )

            # Verify response quality
            assert "MCP" in response or "Model Context Protocol" in response
            assert len(sources) > 0
            assert "MCP" in sources[0]["text"]

    def test_follow_up_question_workflow(self, mock_config):
        """Test workflow for follow-up questions using conversation history"""
        with patch('rag_system.AIGenerator') as mock_ai_gen_class:
            mock_ai_gen = Mock()
            mock_ai_gen_class.return_value = mock_ai_gen

            mock_ai_gen.generate_response.side_effect = [
                "Claude is an AI assistant created by Anthropic.",
                "The previous question was about Claude, an AI assistant."
            ]

            rag = RAGSystem(mock_config)
            session_id = "conversation-456"

            # First question
            response1, _ = rag.query("What is Claude?", session_id=session_id)

            # Follow-up question
            response2, _ = rag.query("What was my previous question?", session_id=session_id)

            # Second response should reference history
            assert "Claude" in response2 or "previous question" in response2

            # Verify history was used
            second_call = mock_ai_gen.generate_response.call_args_list[1]
            assert second_call[1]["conversation_history"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
