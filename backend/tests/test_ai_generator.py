"""
Unit tests for AIGenerator tool calling functionality
Tests whether AIGenerator correctly invokes tools and handles responses
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator


class MockContentBlock:
    """Mock for Anthropic API content block"""
    def __init__(self, block_type, text=None, tool_name=None, tool_input=None, block_id=None):
        self.type = block_type
        self.text = text
        self.name = tool_name
        self.input = tool_input
        self.id = block_id


class MockResponse:
    """Mock for Anthropic API response"""
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator's tool calling functionality"""

    def test_generate_response_without_tools(self):
        """Test basic response generation without tool usage"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Setup mock client
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Mock response without tool use
            mock_response = MockResponse(
                content=[MockContentBlock("text", text="This is a direct answer")],
                stop_reason="end_turn"
            )
            mock_client.messages.create.return_value = mock_response

            # Create generator and test
            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(query="What is AI?")

            # Verify response
            assert result == "This is a direct answer"
            mock_client.messages.create.assert_called_once()

            # Verify no tools were passed
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tools" not in call_kwargs

    def test_generate_response_with_tool_use(self):
        """Test response generation when tool is invoked"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            # Setup mock client
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # First response: tool use requested
            initial_response = MockResponse(
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "MCP basics"},
                        block_id="tool_123"
                    )
                ],
                stop_reason="tool_use"
            )

            # Second response: final answer after tool execution
            final_response = MockResponse(
                content=[MockContentBlock("text", text="Based on the course content, MCP stands for Model Context Protocol")],
                stop_reason="end_turn"
            )

            mock_client.messages.create.side_effect = [initial_response, final_response]

            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "[MCP Course - Lesson 1]\nMCP is the Model Context Protocol..."

            # Create tools definition
            tools = [{
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }]

            # Test
            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(
                query="What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="MCP basics"
            )

            # Verify final response
            assert "Based on the course content" in result
            assert "MCP stands for Model Context Protocol" in result

            # Verify two API calls were made
            assert mock_client.messages.create.call_count == 2

    def test_tool_execution_with_error_result(self):
        """Test handling when tool execution returns an error"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Initial response with tool use
            initial_response = MockResponse(
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "nonexistent", "course_name": "Fake Course"},
                        block_id="tool_123"
                    )
                ],
                stop_reason="tool_use"
            )

            # Final response acknowledging error
            final_response = MockResponse(
                content=[MockContentBlock("text", text="I couldn't find that course. Please check the course name.")],
                stop_reason="end_turn"
            )

            mock_client.messages.create.side_effect = [initial_response, final_response]

            # Mock tool manager returning error
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "No course found matching 'Fake Course'"

            tools = [{
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }]

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(
                query="What is in Fake Course?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify error was passed to tool
            mock_tool_manager.execute_tool.assert_called_once()

            # Verify AI acknowledged the error
            assert "couldn't find" in result or "check the course" in result

    def test_multiple_tool_calls_in_response(self):
        """Test handling of multiple tool calls in a single response"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Response with multiple tool calls
            initial_response = MockResponse(
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "topic A"},
                        block_id="tool_1"
                    ),
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "topic B"},
                        block_id="tool_2"
                    )
                ],
                stop_reason="tool_use"
            )

            final_response = MockResponse(
                content=[MockContentBlock("text", text="Both topics are covered")],
                stop_reason="end_turn"
            )

            mock_client.messages.create.side_effect = [initial_response, final_response]

            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = [
                "[Course A] Topic A content",
                "[Course B] Topic B content"
            ]

            tools = [{
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }]

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(
                query="Tell me about topic A and B",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify both tools were called
            assert mock_tool_manager.execute_tool.call_count == 2
            assert "Both topics are covered" in result

    def test_conversation_history_included(self):
        """Test that conversation history is properly included in API call"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            mock_response = MockResponse(
                content=[MockContentBlock("text", text="Continuing from previous context")],
                stop_reason="end_turn"
            )
            mock_client.messages.create.return_value = mock_response

            history = "User: What is MCP?\nAssistant: MCP is Model Context Protocol."

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(
                query="Tell me more about it",
                conversation_history=history
            )

            # Verify history was included in system prompt
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "Previous conversation:" in call_kwargs["system"]
            assert "What is MCP?" in call_kwargs["system"]

    def test_tool_choice_auto_when_tools_provided(self):
        """Test that tool_choice is set to auto when tools are provided"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            mock_response = MockResponse(
                content=[MockContentBlock("text", text="Answer without using tools")],
                stop_reason="end_turn"
            )
            mock_client.messages.create.return_value = mock_response

            tools = [{
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }]

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            generator.generate_response(
                query="What is 2+2?",
                tools=tools,
                tool_manager=Mock()
            )

            # Verify tool_choice was set
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "tool_choice" in call_kwargs
            assert call_kwargs["tool_choice"]["type"] == "auto"

    def test_no_tool_manager_provided_with_tool_use(self):
        """Test behavior when tool use is requested but no tool_manager provided"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Response requesting tool use
            tool_response = MockResponse(
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "test"},
                        block_id="tool_123"
                    )
                ],
                stop_reason="tool_use"
            )
            mock_client.messages.create.return_value = tool_response

            tools = [{
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }]

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")

            # This should not crash, but won't execute tools
            # The method returns the first content block's text, which doesn't exist for tool_use
            # This might be a bug we'll discover!
            try:
                result = generator.generate_response(
                    query="Search something",
                    tools=tools,
                    tool_manager=None  # No tool manager provided
                )
                # If we get here, check what happened
                # The code should have returned None or empty or raised an error
            except (AttributeError, IndexError) as e:
                # Expected - this reveals a bug
                pass

    def test_api_parameters_configured_correctly(self):
        """Test that API parameters (model, temperature, max_tokens) are set correctly"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            mock_response = MockResponse(
                content=[MockContentBlock("text", text="Test response")],
                stop_reason="end_turn"
            )
            mock_client.messages.create.return_value = mock_response

            generator = AIGenerator(api_key="test-key", model="claude-3-opus")
            generator.generate_response(query="Test")

            # Verify parameters
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "claude-3-opus"
            assert call_kwargs["temperature"] == 0
            assert call_kwargs["max_tokens"] == 800


class TestAIGeneratorEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_query(self):
        """Test handling of empty query"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            mock_response = MockResponse(
                content=[MockContentBlock("text", text="Please provide a question")],
                stop_reason="end_turn"
            )
            mock_client.messages.create.return_value = mock_response

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(query="")

            assert len(result) > 0  # Should still return something

    def test_tool_result_empty_string(self):
        """Test when tool returns empty string"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            initial_response = MockResponse(
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_name="search_course_content",
                        tool_input={"query": "test"},
                        block_id="tool_123"
                    )
                ],
                stop_reason="tool_use"
            )

            final_response = MockResponse(
                content=[MockContentBlock("text", text="No information found")],
                stop_reason="end_turn"
            )

            mock_client.messages.create.side_effect = [initial_response, final_response]

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = ""  # Empty result

            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {"type": "object", "properties": {}, "required": []}}]

            generator = AIGenerator(api_key="test-key", model="claude-3-sonnet")
            result = generator.generate_response(
                query="Search",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Should handle empty tool result gracefully
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
