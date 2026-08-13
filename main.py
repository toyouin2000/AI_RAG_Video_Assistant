import uuid

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")

    # ---------------------------------------------------------
    # 1. Generate a unique ID for this video
    # ---------------------------------------------------------
    video_id = str(uuid.uuid4())

    print(f"Video ID: {video_id}")

    # ---------------------------------------------------------
    # 2. Process video/audio input
    # ---------------------------------------------------------
    chunks = process_input(source)

    # ---------------------------------------------------------
    # 3. Transcribe using Whisper
    # ---------------------------------------------------------
    transcript = transcribe_all(chunks, language)

    print(
        f"Raw transcription (first 300 characters): "
        f"{transcript[:300]}"
    )

    # ---------------------------------------------------------
    # 4. Generate title
    # ---------------------------------------------------------
    title = generate_title(transcript)

    # ---------------------------------------------------------
    # 5. Generate summary
    # ---------------------------------------------------------
    summary = summarize(transcript)

    # ---------------------------------------------------------
    # 6. Extract action items
    # ---------------------------------------------------------
    action_items = extract_action_items(transcript)

    # ---------------------------------------------------------
    # 7. Extract key decisions
    # ---------------------------------------------------------
    decisions = extract_key_decisions(transcript)

    # ---------------------------------------------------------
    # 8. Extract open questions
    # ---------------------------------------------------------
    questions = extract_questions(transcript)

    # ---------------------------------------------------------
    # 9. Build RAG chain for THIS video only
    #
    # IMPORTANT:
    # video_id is passed to the RAG layer so that the
    # transcript can be isolated from other videos.
    # ---------------------------------------------------------
    rag_chain = build_rag_chain(
        transcript=transcript,
        video_id=video_id,
    )

    # ---------------------------------------------------------
    # 10. Return all information
    # ---------------------------------------------------------
    return {
        "video_id": video_id,
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    # ---------------------------------------------------------
    # CLI entry point
    # ---------------------------------------------------------

    source = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    language = (
        input(
            "Language (english/hinglish): "
        ).strip()
        or "english"
    )

    result = run_pipeline(
        source=source,
        language=language,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)

    print(f"🎥 Video ID: {result['video_id']}")

    print(f"\n📌 Title: {result['title']}")

    print(
        f"\n📋 Summary:\n"
        f"{result['summary']}"
    )

    print(
        f"\n✅ Action Items:\n"
        f"{result['action_items']}"
    )

    print(
        f"\n🔑 Key Decisions:\n"
        f"{result['key_decisions']}"
    )

    print(
        f"\n❓ Open Questions:\n"
        f"{result['open_questions']}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Phase 2 — Chat with THIS video
    # ---------------------------------------------------------

    print(
        "\n💬 Chat with this video "
        "(type 'exit' to quit)\n"
    )

    rag_chain = result["rag_chain"]

    while True:

        question = input("You: ").strip()

        if question.lower() in [
            "exit",
            "quit",
            "q",
        ]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(
            rag_chain,
            question,
        )

        print(
            f"\n🤖 Assistant: {answer}\n"
        )