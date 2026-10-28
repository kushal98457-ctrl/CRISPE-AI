import sys
import pytest

def test_python_version():
    """Verify runtime Python interpreter is at least 3.12."""
    assert sys.version_info >= (3, 12), f"Python version {sys.version} is lower than 3.12"

def test_core_package_imports():
    """Verify all critical backend packages are importable."""
    import fastapi
    import pydantic
    import google.genai
    import sqlalchemy
    import psycopg
    import pgvector
    import fastmcp
    import dspy
    import gepa
    import litellm

    assert pydantic.__version__.startswith("2."), "Pydantic must be version 2.x"
    assert hasattr(fastapi, "FastAPI"), "FastAPI class must be available"
    assert hasattr(google.genai, "Client"), "Google GenAI Client must be available"

def test_pgvector_sqlalchemy_type():
    """Verify pgvector's SQLAlchemy VECTOR type supports 768 dimensions."""
    from pgvector.sqlalchemy import VECTOR
    vec_type = VECTOR(768)
    assert vec_type.dim == 768

@pytest.mark.asyncio
async def test_fastmcp_instantiation():
    """Verify FastMCP instance initialization and tool schema registration."""
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field

    mcp = FastMCP("test-server")

    class ToolInput(BaseModel):
        prompt: str = Field(description="Sample input prompt")

    @mcp.tool()
    def sample_crispe_tool(input_data: ToolInput) -> str:
        return f"Echo: {input_data.prompt}"

    tool = await mcp.get_tool("sample_crispe_tool")
    assert tool is not None
    assert tool.name == "sample_crispe_tool"
