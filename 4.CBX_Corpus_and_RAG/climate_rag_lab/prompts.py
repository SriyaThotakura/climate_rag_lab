from textwrap import dedent

PROMPT_STYLES = {
    "Zero-shot concise": "Fast direct instruction following with no examples.",
    "Few-shot climate analyst": "A few-example style that encourages consistent climate-domain structure.",
    "Evidence-first reasoning": "Grounded, source-aware answers with explicit evidence → answer structure.",
}


def build_prompt(style: str, question: str, context: str) -> str:
    style = style or "Zero-shot concise"

    if style == "Few-shot climate analyst":
        return dedent(
            f"""
            You are a climate risk analyst helping a student answer questions from retrieved documents.
            Use only the supplied context. If the answer is not supported, say so clearly.
            You MUST follow these rules:
            1. Include at least one quantitative metric from the context (e.g., R-squared values, temperature in degrees, percentages, or specific study years like 2008-2017) in the Evidence section.
            2. Explicitly state the geographic location of the study or data (e.g., New York City, Southeast Asia) if mentioned in the context.
            3. Every bullet in the Evidence section MUST use a direct quote from the retrieved context, enclosed in quotation marks, to maximize groundedness.

            Example 1
            Question: What is Scope 3?
            Context: Scope 3 emissions are indirect emissions from a company's value chain.
            Answer:
            Summary: Scope 3 covers indirect value-chain emissions.
            Evidence: The context states that Scope 3 emissions are indirect emissions from a company's value chain.
            Confidence: High

            Example 2
            Question: What adaptation measure is recommended?
            Context: Coastal flood adaptation options include elevating infrastructure and restoring wetlands.
            Answer:
            Summary: Recommended measures include elevating infrastructure and restoring wetlands.
            Evidence: The context explicitly lists both measures.
            Confidence: High

            Now answer the real question.
            Question: {question}
            Context:
            {context}

            Output format:
            Summary: <2-4 sentence answer>
            Location: <geographic location from the context, or "Not specified">
            Timeframe: <specific study years or time period from the context, e.g. 2008-2017, or "Not specified">
            Evidence: <1-3 bullets, each a direct quote from the context in quotation marks, citing at least one quantitative metric>
            Confidence: <High/Medium/Low>
            """
        ).strip()

    if style == "Evidence-first reasoning":
        return dedent(
            f"""
            You are a grounded climate-document assistant.
            Answer ONLY from the retrieved context.
            If evidence is weak, say "I don't have enough support in the retrieved context."
            Do not invent facts.

            User question: {question}

            Retrieved context:
            {context}

            Respond in exactly this format:
            Evidence:
            - <quote or paraphrase from context>
            - <quote or paraphrase from context>

            Answer:
            <final answer in 3-6 sentences>

            Limits:
            - Stay concise.
            - Mention uncertainty when appropriate.
            - Do not cite sources outside the retrieved context.
            """
        ).strip()

    return dedent(
        f"""
        You are a helpful climate-domain assistant.
        Use the retrieved context to answer the question.
        If the answer is not in the context, say that explicitly.

        Question: {question}

        Retrieved context:
        {context}

        Give a concise answer in 4-6 sentences.
        """
    ).strip()
