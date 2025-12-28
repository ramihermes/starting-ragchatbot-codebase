# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system that enables semantic search and AI-powered Q&A over educational course content. The system uses ChromaDB for vector storage, Anthropic's Claude API for generation, and provides a web interface via FastAPI.

## Development Commands

### Running the Application
```bash
# Quick start (from root)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application runs at `http://localhost:8000` with API docs at `/docs`.

### Package Management
```bash
# Install/sync dependencies
uv sync

# Add a new dependency
uv add <package-name>
```

### Environment Setup
Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Architecture

### Core Data Flow

The system uses a **two-collection architecture** in ChromaDB:

1. **`course_catalog` collection**: Stores course metadata (titles, instructors, course links) for semantic course name resolution
2. **`course_content` collection**: Stores chunked lesson content with metadata (course_title, lesson_number, chunk_index)

### Query Processing Pipeline

1. User submits query → FastAPI endpoint (`/api/query`)
2. RAGSystem orchestrates the process:
   - AIGenerator receives query + tool definitions
   - Claude decides whether to use the `search_course_content` tool
   - If searching:
     - CourseSearchTool calls VectorStore.search()
     - VectorStore first resolves course name (if provided) via semantic search on `course_catalog`
     - Then searches `course_content` with filters
   - Tool results returned to Claude for synthesis
3. Response returned with sources list

### Component Responsibilities

**RAGSystem** (`rag_system.py`): Main orchestrator
- Initializes all components
- Manages document ingestion pipeline
- Coordinates query → response flow
- Handles session management

**VectorStore** (`vector_store.py`): Vector database interface
- Manages two ChromaDB collections
- Handles semantic course name resolution via `_resolve_course_name()`
- Executes filtered content searches
- Uses sentence-transformers for embeddings

**DocumentProcessor** (`document_processor.py`): Text processing
- Parses structured course documents (expected format: Course Title/Link/Instructor headers, then Lesson markers)
- Chunks text with sentence-aware splitting and overlap
- Creates CourseChunk objects with metadata

**AIGenerator** (`ai_generator.py`): Claude API wrapper
- Implements tool calling flow
- Handles tool execution via `_handle_tool_execution()`
- Manages conversation context

**CourseSearchTool** (`search_tools.py`): Search implementation
- Implements Tool interface for Claude tool calling
- Formats search results with course/lesson context
- Tracks sources for UI display

### Configuration

All settings are in `config.py` (backed by `.env`):
- `CHUNK_SIZE`: 800 characters (size of text chunks)
- `CHUNK_OVERLAP`: 100 characters (overlap between chunks)
- `MAX_RESULTS`: 5 (number of search results)
- `MAX_HISTORY`: 2 (conversation turns to remember)
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"

### Document Format

Course documents in `docs/` follow this structure:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]

Lesson 1: [Lesson Title]
...
```

### Key Implementation Details

**Duplicate Prevention**: When adding courses, the system checks `existing_course_titles` from the vector store to avoid re-processing documents on startup.

**Chunk Context**: First chunks of lessons get prefixed with "Lesson X content:", subsequent chunks in final lesson get "Course [title] Lesson X content:" for better retrieval.

**Tool Calling**: The system uses Anthropic's tool calling API. Claude decides when to search, executes the tool, and synthesizes results into natural language responses.

**Session Management**: Conversations are tracked by session_id, with configurable history length (MAX_HISTORY).

## API Endpoints

- `POST /api/query`: Submit questions (body: `{query, session_id?}`)
- `GET /api/courses`: Get course statistics
- `GET /`: Web interface (served from `frontend/`)
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files