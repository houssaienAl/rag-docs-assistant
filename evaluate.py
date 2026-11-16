"""Evaluate the pipeline on eval/questions.json.

  python evaluate.py              retrieval only: hit@1, hit@k, MRR
  python evaluate.py --generate   also generate answers and check the expected keyword
"""
import json
import sys

import rag


def test_chunk():
    text = "\n\n".join(f"para {i} " + "x" * 300 for i in range(6)) + "\n\n" + "y" * 2000
    chunks = rag.chunk(text, size=800, overlap=1)
    assert all(len(c) <= 800 + 400 for c in chunks), [len(c) for c in chunks]
    assert "para 0" in chunks[0] and "y" * 100 in chunks[-1]
    assert any("para 1" in a and "para 1" in b for a, b in zip(chunks, chunks[1:])), "overlap missing"
    assert rag.chunk("") == []


def main(generate=False):
    test_chunk()
    cases = json.load(open("eval/questions.json", encoding="utf-8"))
    hit1 = hitk = mrr = correct = 0
    for case in cases:
        hits = rag.retrieve(case["question"])
        ranks = [i for i, h in enumerate(hits, 1) if h["source"] == case["source"]]
        hit1 += ranks[:1] == [1]
        hitk += bool(ranks)
        mrr += 1 / ranks[0] if ranks else 0
        line = f"{'ok ' if ranks else 'MISS'} rank={ranks[0] if ranks else '-'}  {case['question']}"
        if generate:
            answer = "".join(rag.answer_stream(case["question"], hits))
            good = case["keyword"].lower() in answer.lower()
            correct += good
            line += f"\n      answer {'ok' if good else 'WRONG'}: {answer.strip()[:160]!r}"
        print(line)
    n = len(cases)
    print(f"\nretrieval over {n} questions: hit@1={hit1 / n:.0%}  hit@{rag.TOP_K}={hitk / n:.0%}  MRR={mrr / n:.2f}")
    if generate:
        print(f"answers containing the expected fact: {correct}/{n} ({correct / n:.0%}) with {rag.LLM_MODEL}")


if __name__ == "__main__":
    main(generate="--generate" in sys.argv)
