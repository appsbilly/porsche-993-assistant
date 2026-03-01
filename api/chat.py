"""
Porsche 993 RAG Chat Engine.
Takes a question -> searches Pinecone vector DB -> sends relevant context to Claude -> returns answer.

Usage:
    python chat.py "My 993 has a rough idle after warming up"
    python chat.py  # Interactive mode
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

INDEX_NAME = "porsche-993"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many chunks to retrieve per query
TOP_K = 15

# Porsche part number pattern: 993.116.015.04 or 993-116-015-04
PART_NUMBER_RE = re.compile(r'\b(\d{3}[\.\-]\d{3}[\.\-]\d{3}[\.\-]\d{2})\b')

# Parts suppliers with search URLs
PARTS_SUPPLIERS = [
    ("Pelican Parts", "https://www.pelicanparts.com/cgi-bin/smart_search.cgi?keywords={}"),
    ("FCP Euro", "https://www.fcpeuro.com/parts?keyword={}"),
    ("Design 911", "https://www.design911.co.uk/search/?q={}"),
]


# ---------------------------------------------------------------------------
# Dynamic system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(car_profile: dict | None = None) -> str:
    """Build system prompt dynamically from the user's car profile."""
    if car_profile:
        year = car_profile.get("year", "")
        model = car_profile.get("model", "993")
        transmission = car_profile.get("transmission", "")
        mileage = car_profile.get("mileage", "")
        known_issues = car_profile.get("known_issues", "")

        car_section = (
            f"THE OWNER'S CAR:\n"
            f"- {year} Porsche 911 ({model})\n"
            f"- {transmission} transmission\n"
            f"- Approximately {mileage} miles"
        )
        if known_issues:
            car_section += f"\n- Known issues: {known_issues}"

        advice_lines = []
        ml = (model or "").lower()
        tl = (transmission or "").lower()
        if "targa" in ml:
            advice_lines.append("- Targa-specific issues (roof seal leaks, body flex, Targa top mechanism)")
        if "cabriolet" in ml or "cab" in ml:
            advice_lines.append("- Cabriolet-specific issues (soft top mechanism, hydraulics, rear window)")
        if "tiptronic" in tl:
            advice_lines.append("- Tiptronic-specific advice (fluid changes, shift adaptation, valve body)")
        if "turbo" in ml:
            advice_lines.append("- Turbo-specific advice (boost control, wastegate, intercooler, K24/K16 turbos)")
        if mileage:
            advice_lines.append(f"- Mileage-appropriate maintenance (what's due at {mileage} miles)")
        if advice_lines:
            car_section += (
                "\nAlways tailor your advice to this specific car. For example:\n"
                + "\n".join(advice_lines)
            )
    else:
        car_section = (
            "THE OWNER'S CAR:\n"
            "- Porsche 911 (993)\n"
            "- No specific details provided yet.\n"
            "Give general 993 advice until the owner shares their car details."
        )

    return f"""You are an expert Porsche 993 mechanic and advisor. You help the owner
diagnose problems, perform repairs, and maintain their car.

{car_section}

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
   when available in the sources. Format part numbers in bold so they stand out.
4. When there's disagreement in the forums, mention the different perspectives.
5. Always err on the side of caution with safety-critical repairs (brakes,
   suspension, steering). Recommend professional help when appropriate.
6. Be conversational and practical, like a knowledgeable friend in the garage.
7. You don't need to repeat the car specs back to the owner every time — they
   know what they drive. Just give relevant, tailored advice.
8. When discussing repairs that require parts, always mention the OEM part
   numbers if they appear in the forum knowledge so the owner can order them.
9. When the user shares photos of their car, parts, or issues:
   - Describe what you see in the image clearly and specifically.
   - Identify any parts, damage, wear patterns, leaks, or issues visible.
   - Cross-reference what you see with the forum knowledge provided.
   - If you can identify part numbers or specific components, mention them.
   - Be specific about location and severity of any visible issues.
10. When your source context references repair photos, diagrams, or step-by-step
    images, describe what those images would show so the user understands the
    procedure visually.

When you reference source material, mention the source forum and thread topic
so the user can look it up for more detail."""


# Keep the old constant for backwards compatibility (CLI mode)
SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# Query rewriting for follow-up questions
# ---------------------------------------------------------------------------

def rewrite_follow_up(prompt: str, conversation_history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone search query.

    When the user asks "Should I just replace the part?", the RAG search needs
    to know WHAT part — which comes from the conversation history. This function
    uses Claude Haiku to rewrite follow-ups into self-contained queries.

    Returns the original prompt unchanged if it's already self-contained or
    if there's no conversation history.
    """
    if not conversation_history:
        return prompt

    # Only rewrite if the prompt looks like a follow-up (short, uses pronouns, etc.)
    # But to keep it simple and reliable, always rewrite when there's history
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return prompt

    # Build a compact summary of recent conversation (last 4 messages max)
    recent = conversation_history[-4:]
    conv_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant messages to just the first ~200 chars
        content = msg["content"]
        if role == "User" and msg.get("images"):
            content += f" [attached {len(msg['images'])} photo(s)]"
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "..."
        conv_lines.append(f"{role}: {content}")

    conv_text = "\n".join(conv_lines)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-20250414",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"""Rewrite this follow-up question into a standalone search query that includes the necessary context from the conversation. The query should work for searching a Porsche 993 forum knowledge base.

CONVERSATION:
{conv_text}

FOLLOW-UP QUESTION: {prompt}

Write ONLY the rewritten search query, nothing else. If the question is already self-contained, return it unchanged.""",
            }],
        )
        rewritten = response.content[0].text.strip()
        # Sanity check: don't return something wildly different or too long
        if rewritten and len(rewritten) < 500:
            return rewritten
    except Exception:
        pass  # Fall back to original prompt

    return prompt


# ---------------------------------------------------------------------------
# Parts helpers
# ---------------------------------------------------------------------------

def extract_part_numbers(text: str) -> list[str]:
    """Extract Porsche OEM part numbers from text."""
    return list(set(PART_NUMBER_RE.findall(text)))


def generate_parts_links(part_numbers: list[str]) -> str:
    """Generate markdown links to search for parts on major suppliers."""
    if not part_numbers:
        return ""

    lines = ["\n\n---\n**🛒 Order Parts**"]
    seen = set()
    for pn in part_numbers:
        pn_clean = pn.replace("-", ".").strip()
        if pn_clean in seen:
            continue
        seen.add(pn_clean)
        supplier_links = " · ".join(
            f"[{name}]({url.format(pn_clean)})" for name, url in PARTS_SUPPLIERS
        )
        lines.append(f"- **{pn_clean}**: {supplier_links}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embedding via HuggingFace Inference API
# ---------------------------------------------------------------------------

_index = None


_hf_client = None


def _get_hf_client():
    """Lazy-load the HuggingFace InferenceClient."""
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        api_key = os.getenv("HF_API_KEY", "") or None
        _hf_client = InferenceClient(token=api_key)
    return _hf_client


def _embed_query(text: str) -> list[float]:
    """Get query embedding from the HuggingFace Inference API.

    Uses the same all-MiniLM-L6-v2 model that was used at index-build time,
    so vectors are compatible with existing Pinecone data.

    Uses the huggingface_hub InferenceClient which handles the new
    router.huggingface.co endpoints automatically.
    """
    client = _get_hf_client()

    try:
        result = client.feature_extraction(
            text=text,
            model=EMBEDDING_MODEL,
        )
    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "unauthorized" in err_str:
            raise RuntimeError(
                "HuggingFace API returned 401 Unauthorized. "
                "Your HF_API_KEY may be missing or may need 'Inference Providers' "
                "permission. Create a new token at https://huggingface.co/settings/tokens "
                "with 'Make calls to Inference Providers' enabled."
            ) from e
        raise

    # InferenceClient returns a numpy array or list of shape (dim,) or (1, dim)
    import numpy as np
    if isinstance(result, np.ndarray):
        if result.ndim == 2:
            return result[0].tolist()
        return result.tolist()

    # Fallback: raw list handling
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            return result[0] if len(result) == 1 else result
        return result

    raise ValueError(f"Unexpected embedding response shape: {type(result)}")


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
    index = _get_index()

    # Generate query embedding via HuggingFace API
    query_embedding = _embed_query(query)

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


def _car_description(car_profile: dict | None) -> str:
    """One-line car description for user messages."""
    if not car_profile:
        return "their Porsche 993"
    parts = []
    if car_profile.get("year"):
        parts.append(car_profile["year"])
    parts.append("Porsche 993")
    if car_profile.get("model"):
        parts.append(car_profile["model"])
    if car_profile.get("transmission"):
        parts.append(car_profile["transmission"])
    desc = " ".join(parts)
    if car_profile.get("mileage"):
        desc += f" (~{car_profile['mileage']} miles)"
    return desc


def ask(question: str, verbose: bool = False, car_profile: dict | None = None) -> str:
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
    car_desc = _car_description(car_profile)
    system_prompt = build_system_prompt(car_profile)

    # Format the user message with context
    user_message = f"""Based on the following knowledge from Porsche forums and technical articles,
answer this question about the owner's {car_desc}:

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
        system=system_prompt,
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

    # Append parts links
    part_numbers = extract_part_numbers(answer)
    if part_numbers:
        answer += generate_parts_links(part_numbers)

    return answer


def ask_stream(question: str, verbose: bool = False, car_profile: dict | None = None):
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
    car_desc = _car_description(car_profile)
    system_prompt = build_system_prompt(car_profile)

    user_message = f"""Based on the following knowledge from Porsche forums and technical articles,
answer this question about the owner's {car_desc}:

QUESTION: {question}

FORUM KNOWLEDGE:
{context}

Please provide a helpful, practical answer based on this knowledge. Cite sources
when referencing specific advice. If the sources are insufficient, say so."""

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
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
