"""
Unit tests for CourseSearchTool.execute() method
Tests various scenarios: successful searches, empty results, errors, filtering
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    def test_successful_search_no_filters(self):
        """Test basic search without course or lesson filters"""
        # Setup mock vector store
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Content from lesson 1", "Content from lesson 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ],
            distances=[0.5, 0.6],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Create tool and execute search
        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test query")

        # Verify search was called correctly
        mock_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )

        # Verify result format
        assert "[Test Course - Lesson 1]" in result
        assert "Content from lesson 1" in result
        assert "[Test Course - Lesson 2]" in result
        assert "Content from lesson 2" in result

    def test_successful_search_with_course_filter(self):
        """Test search with course name filter"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Course specific content"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
            distances=[0.4],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="MCP basics", course_name="Introduction to MCP")

        # Verify course filter passed correctly
        mock_store.search.assert_called_once_with(
            query="MCP basics",
            course_name="Introduction to MCP",
            lesson_number=None
        )

        assert "Introduction to MCP" in result
        assert "Course specific content" in result

    def test_successful_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Advanced Topics", "lesson_number": 3}],
            distances=[0.3],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson3"

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="advanced concepts", lesson_number=3)

        # Verify lesson filter passed correctly
        mock_store.search.assert_called_once_with(
            query="advanced concepts",
            course_name=None,
            lesson_number=3
        )

        assert "Lesson 3" in result
        assert "Lesson 3 content" in result

    def test_empty_results_no_filters(self):
        """Test handling of empty search results without filters"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="nonexistent content")

        assert "No relevant content found" in result
        assert "nonexistent content" not in result  # Should not echo the query

    def test_empty_results_with_course_filter(self):
        """Test handling of empty results when course filter is applied"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test", course_name="Nonexistent Course")

        assert "No relevant content found" in result
        assert "in course 'Nonexistent Course'" in result

    def test_empty_results_with_lesson_filter(self):
        """Test handling of empty results when lesson filter is applied"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test", lesson_number=99)

        assert "No relevant content found" in result
        assert "in lesson 99" in result

    def test_error_from_vector_store(self):
        """Test handling of errors from vector store"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: Database connection failed"
        )
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test query")

        # Should return the error message
        assert "Search error: Database connection failed" in result

    def test_sources_tracking(self):
        """Test that sources are properly tracked in last_sources"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.5, 0.6],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.side_effect = [
            "https://example.com/courseA/lesson1",
            "https://example.com/courseB/lesson2"
        ]

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        # Verify sources were tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/courseA/lesson1"
        assert tool.last_sources[1]["text"] == "Course B - Lesson 2"
        assert tool.last_sources[1]["url"] == "https://example.com/courseB/lesson2"

    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted for Anthropic API"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)
        definition = tool.get_tool_definition()

        # Verify structure
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Verify schema
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_multiple_searches_reset_sources(self):
        """Test that sources are reset between searches"""
        mock_store = Mock()

        # First search
        mock_results_1 = SearchResults(
            documents=["Content 1"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.5],
            error=None
        )
        mock_store.search.return_value = mock_results_1
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_store)
        tool.execute(query="first query")
        assert len(tool.last_sources) == 1

        # Second search with different results
        mock_results_2 = SearchResults(
            documents=["Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course B", "lesson_number": 2},
                {"course_title": "Course C", "lesson_number": 3}
            ],
            distances=[0.4, 0.6],
            error=None
        )
        mock_store.search.return_value = mock_results_2
        mock_store.get_lesson_link.side_effect = [
            "https://example.com/lesson2",
            "https://example.com/lesson3"
        ]

        tool.execute(query="second query")

        # Sources should be from second search only
        assert len(tool.last_sources) == 2
        assert "Course B" in tool.last_sources[0]["text"]
        assert "Course C" in tool.last_sources[1]["text"]


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_and_execute_tool(self):
        """Test registering and executing a tool"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.5],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Execute via manager
        result = manager.execute_tool("search_course_content", query="test")

        assert "Test content" in result
        mock_store.search.assert_called_once()

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        mock_store = Mock()
        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_get_last_sources(self):
        """Test retrieving sources from last search"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.5],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Execute search
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "Course - Lesson 1"

    def test_reset_sources(self):
        """Test resetting sources after retrieval"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.5],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        manager = ToolManager()
        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Execute and get sources
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()
        assert len(sources) == 1

        # Reset and verify empty
        manager.reset_sources()
        sources = manager.get_last_sources()
        assert len(sources) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
