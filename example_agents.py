"""
example_agents.py

A minimal, generic Extractor -> Analyzer -> Writer -> Reviewer pipeline
definition, given here as a WORKING EXAMPLE of the shape pipeline.py
expects -- (stage_name, prompt_fn, context_key) tuples, where each
prompt_fn takes `snippet` (the question/input text) plus the running
`context` dict (each prior stage's own output, keyed by that stage's
context_key) and returns the prompt string for this stage.

Replace this file (or add a new one with the same shape and point
pipeline.py at it) with your own team's actual agent stages/prompts --
nothing about Module 1, Module 2, or the two hooks in hooks.py depends on
what these specific prompts say, only on the (stage_name, prompt_fn,
context_key) shape and the sequential-context-passing pattern.
"""

STAGE_NAMES = ("extractor", "analyzer", "writer", "reviewer")


def extractor_prompt(snippet, **_):
    return (
        "Identify what type of question this is (math, logic, trivia, "
        "code, or factual) and restate exactly what answer is being asked "
        "for. Answer in 1-2 short lines, no extra commentary.\n\n"
        f"Question: '{snippet}'"
    )


def analyzer_prompt(snippet, extraction, **_):
    return (
        "Given this question and how it was framed below, outline the key "
        "reasoning steps or approach needed to answer it, in 2-3 short "
        "lines.\n\n"
        f"Question: '{snippet}'\n"
        f"Framing:\n{extraction}"
    )


def writer_prompt(snippet, analysis, **_):
    return (
        "Using the reasoning approach below, write the final answer to "
        "this question. Be direct and concise -- give the answer itself, "
        "not a restatement of the approach.\n\n"
        f"Question: '{snippet}'\n"
        f"Approach:\n{analysis}"
    )


def reviewer_prompt(snippet, writer_output, **_):
    return (
        "Review this answer against the original question for correctness "
        "and clarity. Return the corrected final answer if anything is "
        "wrong, or the same answer unchanged if it is already correct. "
        "Return only the final answer, nothing else.\n\n"
        f"Question: '{snippet}'\n"
        f"Draft answer: '{writer_output}'"
    )


# (stage_name, prompt_fn, context_key) -- context_key is the name each
# stage's own output is stored under in the running `context` dict, so the
# next stage's prompt_fn can pull it out by keyword (see writer_prompt's
# `analysis` param, reviewer_prompt's `writer_output` param) -- identical
# shape to agents.py's PIPELINE_STAGES.
PIPELINE_STAGES = (
    ("extractor", extractor_prompt, "extraction"),
    ("analyzer", analyzer_prompt, "analysis"),
    ("writer", writer_prompt, "writer_output"),
    ("reviewer", reviewer_prompt, "review"),
)
