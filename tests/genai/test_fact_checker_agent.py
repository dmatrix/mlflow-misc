"""
Tests for the Fact-Checker Agent.

These tests verify the basic functionality without requiring an OpenAI API key.
"""

import pytest
from unittest.mock import MagicMock, patch
from genai.agents.fact_checker_agent import (
    web_search,
    check_knowledge_base,
    extract_entities,
    FactCheckerAgent,
    TOOLS,
    AVAILABLE_TOOLS,
)


class TestTools:
    """Test individual tools."""

    def test_web_search(self):
        """Test web search tool returns results."""
        result = web_search("MLflow")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "MLflow" in result or "mlflow" in result.lower()

    def test_check_knowledge_base(self):
        """Test knowledge base lookup."""
        # Test known fact
        result = check_knowledge_base("Python was created by Guido van Rossum")
        assert isinstance(result, dict)
        assert "verified" in result
        assert "confidence" in result
        assert "source" in result
        assert "message" in result

    def test_check_knowledge_base_unknown(self):
        """Test knowledge base with unknown claim."""
        result = check_knowledge_base("Some completely unknown claim about xyz")
        assert isinstance(result, dict)
        assert result["verified"] is False
        assert result["confidence"] == 0.0

    def test_extract_entities(self):
        """Test entity extraction."""
        result = extract_entities("Python was created by Guido van Rossum")
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("Python" in entity for entity in result)

    def test_extract_entities_generic(self):
        """Test entity extraction with generic text."""
        result = extract_entities("Some random text without specific entities")
        assert isinstance(result, list)
        assert len(result) > 0  # Should at least return generic entity


class TestToolDefinitions:
    """Test tool configuration for OpenAI."""

    def test_tools_structure(self):
        """Test that TOOLS list is properly formatted."""
        assert isinstance(TOOLS, list)
        assert len(TOOLS) == 3

        for tool in TOOLS:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_available_tools_mapping(self):
        """Test that all tools are mapped correctly."""
        assert len(AVAILABLE_TOOLS) == 3
        assert "web_search" in AVAILABLE_TOOLS
        assert "check_knowledge_base" in AVAILABLE_TOOLS
        assert "extract_entities" in AVAILABLE_TOOLS

        # Verify they're callable
        for name, func in AVAILABLE_TOOLS.items():
            assert callable(func)


class TestFactCheckerAgent:
    """Test FactCheckerAgent class."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_agent_initialization(self):
        """Test agent initialization with mocked API key."""
        agent = FactCheckerAgent(
            model="gpt-4o-mini", max_iterations=5, temperature=0.1
        )

        assert agent.model == "gpt-4o-mini"
        assert agent.max_iterations == 5
        assert agent.temperature == 0.1
        assert agent.system_prompt is not None
        assert len(agent.system_prompt) > 0

    def test_agent_initialization_no_api_key(self):
        """Test that agent fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                FactCheckerAgent()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_extract_verdict_true(self):
        """Test verdict extraction for TRUE."""
        agent = FactCheckerAgent()

        content = "VERDICT: TRUE\n\nThe claim is accurate based on evidence."
        verdict = agent._extract_verdict(content)
        assert verdict == "TRUE"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_extract_verdict_false(self):
        """Test verdict extraction for FALSE."""
        agent = FactCheckerAgent()

        content = "VERDICT: FALSE\n\nThe claim is incorrect."
        verdict = agent._extract_verdict(content)
        assert verdict == "FALSE"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_extract_verdict_partially_true(self):
        """Test verdict extraction for PARTIALLY TRUE."""
        agent = FactCheckerAgent()

        content = "VERDICT: PARTIALLY TRUE\n\nSome parts are accurate."
        verdict = agent._extract_verdict(content)
        assert verdict == "PARTIALLY TRUE"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_extract_verdict_unverifiable(self):
        """Test verdict extraction for UNVERIFIABLE."""
        agent = FactCheckerAgent()

        content = "VERDICT: UNVERIFIABLE\n\nInsufficient evidence."
        verdict = agent._extract_verdict(content)
        assert verdict == "UNVERIFIABLE"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        agent = FactCheckerAgent()

        result = agent._execute_tool("web_search", {"query": "test"})
        assert isinstance(result, str)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_execute_tool_unknown(self):
        """Test execution of unknown tool."""
        agent = FactCheckerAgent()

        result = agent._execute_tool("unknown_tool", {})
        assert isinstance(result, dict)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_execute_tool_error_handling(self):
        """Test tool execution with invalid arguments."""
        agent = FactCheckerAgent()

        # This should handle the error gracefully
        result = agent._execute_tool("web_search", {"invalid_arg": "test"})
        assert isinstance(result, dict)
        assert "error" in result


class TestIntegration:
    """Integration tests (require mocking OpenAI API)."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("genai.agents.fact_checker_agent.OpenAI")
    def test_verify_claim_mock(self, mock_openai):
        """Test verify_claim with mocked OpenAI response."""
        # Mock the OpenAI response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create a mock response with no tool calls (direct answer)
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "VERDICT: TRUE\n\nThis claim is verified."

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        # Create agent and verify claim
        agent = FactCheckerAgent()
        result = agent.verify_claim("Test claim")

        # Verify result structure
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "reasoning" in result
        assert "iterations" in result
        assert "claim" in result

        assert result["verdict"] == "TRUE"
        assert result["claim"] == "Test claim"
        assert result["iterations"] == 1

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("genai.agents.fact_checker_agent.OpenAI")
    def test_verify_claim_with_tools_mock(self, mock_openai):
        """Test verify_claim with tool calls (mocked)."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First response: agent wants to use a tool
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = '{"query": "test"}'

        mock_message_1 = MagicMock()
        mock_message_1.tool_calls = [mock_tool_call]
        mock_message_1.content = None

        mock_choice_1 = MagicMock()
        mock_choice_1.message = mock_message_1

        mock_response_1 = MagicMock()
        mock_response_1.choices = [mock_choice_1]

        # Second response: agent provides final answer
        mock_message_2 = MagicMock()
        mock_message_2.tool_calls = None
        mock_message_2.content = "VERDICT: TRUE\n\nVerified via web search."

        mock_choice_2 = MagicMock()
        mock_choice_2.message = mock_message_2

        mock_response_2 = MagicMock()
        mock_response_2.choices = [mock_choice_2]

        # Set up the mock to return different responses
        mock_client.chat.completions.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        # Create agent and verify claim
        agent = FactCheckerAgent()
        result = agent.verify_claim("Test claim")

        # Verify the agent made 2 iterations
        assert result["iterations"] == 2
        assert result["verdict"] == "TRUE"

        # Verify OpenAI was called twice
        assert mock_client.chat.completions.create.call_count == 2
