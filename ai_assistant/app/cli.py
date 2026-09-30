"""
Simple terminal chat (Task 1 demo, no web server needed).

    python -m app.cli
"""
import asyncio
import json
import logging

from app.assistant import answer_question
from app.rag import vector_store
from app.rag.ingest import ingest_documents


async def main():
    logging.basicConfig(level=logging.WARNING)

    if vector_store.count_chunks() == 0:
        print("Vector DB is empty, ingesting documents first...")
        print(ingest_documents())

    print("Acme Assistant (type 'exit' to quit)\n")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            continue

        result = await answer_question(question, history=history)
        print("\nAssistant (JSON):")
        print(json.dumps(result, indent=2))
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    asyncio.run(main())
