# NOTE: the prompt separation is partially done for efficiency; previously I tried
# to do it all in one prompt with sequential format() calls but this will cause a backend
# error when the document contains any {} as python will expect the {} to be filled by
# format() arguments

# ruff: noqa: E501, W605 start

# ── Generic document prompts ────────────────────────────────────────────────
CONTEXTUAL_RAG_PROMPT1 = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document"""

CONTEXTUAL_RAG_PROMPT2 = """<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
""".rstrip()

CONTEXTUAL_RAG_TOKEN_ESTIMATE = 64  # 19 + 45

DOCUMENT_SUMMARY_PROMPT = """<document>
{document}
</document>
Please give a short succinct summary of the entire document. Answer only with the succinct summary and nothing else.
""".rstrip()

DOCUMENT_SUMMARY_TOKEN_ESTIMATE = 50

# ── Source-code specific prompts ────────────────────────────────────────────
# These are used instead of the generic prompts when the document is a source
# code file (metadata "type" == "CodeFile").  Including the file path and
# language lets the LLM produce more precise, developer-readable context that
# dramatically improves semantic search on code.
#
# PROMPT1 / PROMPT2 follow the same two-part split as the generic prompts so
# that the cacheable document prefix and the per-chunk suffix can be sent
# separately (see add_chunk_summaries in indexing_pipeline.py).

CODE_CONTEXTUAL_RAG_PROMPT1 = """<code_file path="{file_path}" language="{language}">
{document}
</code_file>
Here is the code section we want to situate within the overall file"""

CODE_CONTEXTUAL_RAG_PROMPT2 = """<code_section>
{chunk}
</code_section>
In 1-2 sentences, describe what this code section does and its role within the file. Mention the function or class name and its primary purpose. Answer only with the succinct context and nothing else.
""".rstrip()

CODE_CONTEXTUAL_RAG_TOKEN_ESTIMATE = 80

CODE_DOCUMENT_SUMMARY_PROMPT = """<code_file path="{file_path}" language="{language}">
{document}
</code_file>
In 2-3 sentences, summarize this source file: its main purpose, the key functions and classes it defines, and what problem it solves. Answer only with the succinct summary and nothing else.
""".rstrip()

CODE_DOCUMENT_SUMMARY_TOKEN_ESTIMATE = 70
# ruff: noqa: E501, W605 end
