"""
llm.py
------
Provides a chat/generation model based on config.settings.llm_provider.

- "huggingface": runs a local generation pipeline. Default model:
  Qwen/Qwen2.5-1.5B-Instruct (free, no HF token/login required, good
  quality-vs-speed balance on CPU). Also supports seq2seq models like the
  flan-t5 family — the encoder-decoder vs. causal model type is
  auto-detected from the model's config, so you can swap HF_LLM_MODEL to
  any compatible model name without touching this code.
- "openai": uses OpenAI's chat completion API (default: gpt-4o-mini). Needs
  OPENAI_API_KEY, but gives noticeably higher-quality answers.
"""

from __future__ import annotations

from config import settings


def get_llm(provider: str | None = None):
    provider = (provider or settings.llm_provider).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add it to your .env file, or switch LLM_PROVIDER to 'huggingface'."
            )
        return ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )

    if provider == "groq":
        from langchain_openai import ChatOpenAI

        if not settings.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set. "
                "Get a free key from https://console.groq.com and add it to your .env file."
            )
        # Groq's API is OpenAI-compatible -- same client, just a different
        # base_url and model name, so ChatOpenAI works unmodified here.
        return ChatOpenAI(
            model=settings.groq_llm_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            temperature=0.2,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            pipeline,
        )

        model_name = settings.hf_llm_model
        model_config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = getattr(model_config, "is_encoder_decoder", False)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        gen_kwargs = dict(
            max_new_tokens=settings.max_new_tokens,
            repetition_penalty=1.15,
            do_sample=False,
        )

        if is_encoder_decoder:
            # Seq2seq family, e.g. flan-t5-*.
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            try:
                pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
            except Exception:
                pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
            return HuggingFacePipeline(pipeline=pipe)

        # Causal/instruct family, e.g. Qwen2.5-Instruct, Phi-3-mini-instruct.
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            **gen_kwargs,
        )
        # return_full_text must be passed here, via pipeline_kwargs, not
        # just at pipeline() construction time above -- LangChain's
        # HuggingFacePipeline re-applies its own call-time kwargs, which
        # otherwise silently override the pipeline's own defaults and
        # cause the input prompt to be echoed back as part of the answer.
        base_llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs={"return_full_text": False})

        if getattr(tokenizer, "chat_template", None):
            # CRITICAL: instruct-tuned causal models (Qwen2.5-Instruct, Phi-3,
            # etc.) are trained on a structured chat format
            # (<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant).
            # Feeding them a raw prompt string bypasses that entirely - the
            # model just treats the prompt as text to *continue* rather than
            # an instruction to follow, which is what produces rambling,
            # associative, off-topic completions even when the retrieved
            # context contains the right answer. ChatHuggingFace applies the
            # tokenizer's chat_template automatically so the model actually
            # sees the format it was fine-tuned on.
            from langchain_huggingface import ChatHuggingFace

            return ChatHuggingFace(llm=base_llm, tokenizer=tokenizer)

        return base_llm

    raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'huggingface', 'openai', or 'groq'.")
