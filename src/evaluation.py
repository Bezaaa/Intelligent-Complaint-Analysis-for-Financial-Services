
from src.rag_pipeline import answer_question

test_questions = [
    "Why are users unhappy with the Buy Now Pay Later service?",
    "What are the most common issues with credit cards?",
    "Do users report fraud with money transfers?",
    "Are there complaints about savings accounts not earning interest?",
]

for q in test_questions:
    print(f"\n\n🧠 Question: {q}")
    answer, sources = answer_question(q)
    print(f"💬 Answer: {answer}")
    print("\n🔍 Top Retrieved Sources:")
    for i, doc in enumerate(sources[:2], 1):
        print(f"  [{i}] {doc.page_content[:300]}...\n")
