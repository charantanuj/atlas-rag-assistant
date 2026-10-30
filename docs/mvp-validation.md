# MVP validation record

Run date: 2026-10-29. Local API, existing corpus, Gemini Flash-Lite. One small smoke run; results are not a benchmark or a proof of correctness.

| Case | Expected | Observed | Time |
|---|---|---|---:|
| Greeting | conversational | timeout | 35.015 s |
| Identity | conversational | conversational | 0.763 s |
| Instruction override | blocked | blocked | 0.007 s |
| Shorthand override | blocked | blocked | 0.009 s |
| Five-level paging | cited answer, paging source | answered, 4 citations | 3.874 s |
| AVX-512 sorting | cited answer, quicksort source | answered, 2 citations | 1.096 s |
| GPU skiplist | cited answer, skiplist source | answered, 3 citations | 0.859 s |
| Imaginary bank password | abstain | insufficient evidence | 2.097 s |
| Repeat AVX-512 | cited answer | answered, 2 citations | 0.008 s |
| Memory consistency | cited answer, memory source | insufficient evidence | 1.471 s |

8 of 10 passed the script's expected status and source-name checks. The repeated question may have been cached from an earlier session; the script does not verify cache-hit metadata. Source-name matching is not semantic evaluation. The first greeting timeout remains unexplained; subsequent fast-path requests succeeded. No latency guarantee is established.

Manual source check: the first page of `A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel Skylake (1704.08579).pdf` reports a 4x speedup against GNU C++ sort and a 1.4x speedup against Intel IPP for the evaluated approach. This supports those specific claims in the tested answer. Other citations and claims have not been manually verified.

The deterministic integration test `tests/integration/test_upload_to_answer.py` exercises text upload, chunking, deterministic embeddings, in-memory retrieval, fake generation, and assembled citations. It does not test a real PDF, Chroma, Gemini, authentication, or end-to-end semantic faithfulness. The latest full backend run passed 120 tests.

To rerun the live smoke cases, start the backend from `backend`, then run `.\.venv\Scripts\python.exe -u scripts/evaluate_mvp.py`. This sends questions and retrieved context to Gemini and may incur API usage charges. Never print or commit credentials.
