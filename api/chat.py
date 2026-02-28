"""
Porsche 993 RAG Chat Engine.
Takes a question -> searches Pinecone vector DB -> sends relevant context to Claude -> returns answer.

Usage:
    python chat.py "My 993 has a rough idle after warming up"
    python chat.py  # Interactive mode
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

INDEX_NAME = "porsche-993"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many chunks to retrieve per query
TOP_K = 15

# System prompt for Claude
SYSTEM_PROMPT = """You are an expert Porsche 993 mechanic and advisor. You help the owner
diagnose problems, perform repairs, and maintain their car.

THE OWNER'S CAR:
- 1997 Porsche 911 (993) Targa
- Tiptronic (automatic) transmission
- Approximately 80,000 miles
Always tailor your advice to this specific car. For example:
- Targa-specific issues (roof seal leaks, body flex, Targa top mechanism)
- Tiptronic-specific advice (fluid changes, shift adaptation, valve body)
- Mileage-appropriate maintenance (what's due at 80k, what to watch for)

You have access to real knowledge from Porsche forums (Pelican Parts, Rennlist, 911uk,
6SpeedOnline, TIPEC, Carpokes) and technical articles, DIY guides, and YouTube transcripts
from experienced 993 owners and mechanics.

RULES:
1. Base your answers on the provided forum knowledge. If the sources contain
   relevant information, cite it naturally (e.g. "According to a Rennlist thread..."
   or "A Pelican Parts tech article explains...").
2. If the sources don't contain enough info to fully answer, say so honestly
   and share what you do know from general 993 knowledge.
3. Include specific part numbers, torque specs, and step-by-step procedures
   when available in the sources.
4. When there's disagreement in the forums, mention the different perspectives.
5. Always err on the side of caution with safety-critical repairs (brakes,
   suspension, steering). Recommend professional help when appropriate.
6. Be conversational and practical, like a knowledgeable friend in the garage.
7. You don't need to repeat the car specs back to the owner every time — they
   know what they drive. Just give relevant, tailored advice.

When you reference source material, mention the source forum and thread topic
so the user can look it up for more detail."""

# Cache the embedding model and Pinecone index
_model = None
_index = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_index():
    """Lazy-load the Pinecone index."""
    global _index
    if _index is None:
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not set in .env")
            sys.exit(1)

        pc = Pinecone(api_key=api_key)
        _index = pc.Index(INDEX_NAME)
    return _index


def search(query: str, n_results: int = TOP_K) -> list[dict]:
    """Search Pinecone for relevant chunks."""
    model = _get_model()
    index = _get_index()

    # Generate query embedding
    query_embedding = model.encode([query]).tolist()[0]

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True,
    )

    sources = []
    for match in results.matches:
        meta = match.metadata
        sources.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "content_type": meta.get("content_type", ""),
            "relevance": match.score,
        })

    return sources


def build_context(sources: list[dict]) -> str:
    """Format retrieved sources into context for Claude."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        header = f"[Source {i}] {src['title']}"
        if src['source']:
            header += f" ({src['source']})"
        context_parts.append(f"{header}\n{src['text']}")

    return "\n\n" + "=" * 60 + "\n\n".join(context_parts)


def ask(question: str, verbose: bool = False) -> str:
    """Ask a question and get an answer from Claude with forum knowledge."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        return ("❌ Please set your ANTHROPIC_API_KEY in .env\n"
                "   Get one at: https://console.anthropic.com/")

    # Search for relevant context
    sources = search(question)

    if verbose:
        print(f"\n🔍 Found {len(sources)} relevant sources:")
        for s in sources[:5]:
            print(f"   [{s['relevance']:.3f}] {s['title'][:50]} ({s['source']})")
        print()

    # Build the context
    context = build_context(sources)

    # Format the user message with context
    user_message = f"""Based on the following knowledge from Porsche forums and technical articles,
answer this question about the owner's 1997 Porsche 993 Targa Tiptronic (~80k miles):

QUESTION: {question}

FORUM KNOWLEDGE:
{context}

Please provide a helpful, practical answer based on this knowledge. Cite sources
when referencing specific advice. If the sources are insufficient, say so."""

    # Call Claude
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    # Append source links
    unique_urls = []
    seen = set()
    for s in sources[:5]:
        if s["url"] and s["url"] not in seen:
            seen.add(s["url"])
            unique_urls.append(f"  - {s['title'][:60]} — {s['url']}")

    if unique_urls:
        answer += "\n\n📚 Sources:\n" + "\n".join(unique_urls)

    return answer


def ask_stream(question: str, verbose: bool = False):
    """Stream an answer from Claude with forum knowledge. Yields text chunks."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        yield ("❌ Please set your ANTHROPIC_API_KEY in .env\n"
               "   Get one at: https://console.anthropic.com/")
        return

    # Search for relevant context
    sources = search(question)

    if verbose:
        print(f"\n🔍 Found {len(sources)} relevant sources:")
        for s in sources[:5]:
            print(f"   [{s['relevance']:.3f}] {s['title'][:50]} ({s['source']})")
        print()

    context = build_context(sources)

    user_message = f"""Based on the following knowledge from Porsche forums and technical articles,
answer this question about the owner's 1997 Porsche 993 Targa Tiptronic (~80k miles):

QUESTION: {question}

FORUM KNOWLEDGE:
{context}

Please provide a helpful, practical answer based on this knowledge. Cite sources
when referencing specific advice. If the sources are insufficient, say so."""

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text

    # Append source links
    unique_urls = []
    seen = set()
    for s in sources[:5]:
        if s["url"] and s["url"] not in seen:
            seen.add(s["url"])
            unique_urls.append(f"  - {s['title'][:60]} — {s['url']}")

    if unique_urls:
        yield "\n\n📚 Sources:\n" + "\n".join(unique_urls)


def interactive_mode():
    """Run an interactive chat session."""
    print("=" * 60)
    print("🔧 Porsche 993 Repair Assistant")
    print("   Powered by real forum knowledge (Pinecone + Claude)")
    print("   Type 'quit' to exit, 'verbose' to toggle source details")
    print("=" * 60)

    verbose = False

    while True:
        try:
            question = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break
        if question.lower() == "verbose":
            verbose = not verbose
            print(f"   Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        print("\n🔍 Searching forum knowledge...")
        answer = ask(question, verbose=verbose)
        print(f"\n{answer}")


def main():
    """Entry point — CLI argument or interactive mode."""
    # Quick connectivity check
    index = _get_index()
    stats = index.describe_index_stats()
    print(f"✅ Connected to Pinecone — {stats.total_vector_count} knowledge chunks")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n❓ {question}")
        print("\n🔍 Searching forum knowledge...")
        answer = ask(question, verbose=True)
        print(f"\n{answer}")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
