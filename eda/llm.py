import os, json, requests

def call_llm(payload: dict,
             provider: str | None = None,
             model: str | None = None,
             api_key: str | None = None,
             api_base: str | None = None): 
    """
    provider: 'openai' | 'ollama' | 'fake'
    Returns: {"ok"}: bool, "json"" dict|None, "text": str|None, "error": str|None}
    """
    provider = provider or os.getenv("LLM_PROVIDER", "fake")
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    system = "You are a careful data analyst. Use only the provided numbers; do not invent results. Return concise JSON."
    user_msg = {
        "instruction": (
            "Summarize the dataset and list 5–8 next actions. "
            "Return JSON with keys: title, one_line, domain_guess, key_facts[], issues[], actions[], questions[]."
        ),
        "context": payload
    }
    
    try:
        if provider == 'openai': 
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                            base_url=api_base or os.getenv("OPENAI_BASE_URL", None))
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_msg)}
                ],
            )
            txt = resp.choices[0].message.content
            return {"ok": True, "json": json.loads(txt), "text": txt, "error": None}
        
        if provider == "ollama": 
            url = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat")
            r = requests.post(url, json={
                "model": model or "llama3.1:8b",
                "messages": [
                   {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_msg)}
                ],
                "options": {"temperature": 0.2},
                "stream": False
            }, timeout=120)
            r.raise_for_status()
            # Some local models reply with plain text; try JSON parse then fallback
            txt = r.json()["message"]["content"]
            try:
                return {"ok": True, "json": json.loads(txt), "text": txt, "error": None}
            except Exception:
                return {"ok": True, "json": None, "text": txt, "error": None}

        # fallback “fake” (always available)
        return {
            "ok": True,
            "json": {
                "title": "Auto Summary (stub)",
                "one_line": "Contract CSV reviewed; see key facts and next steps.",
                "domain_guess": "generic",
                "key_facts": ["rows, missingness, uniques parsed from contract CSV."],
                "issues": ["This is a stub; switch provider to OpenAI or Ollama for real analysis."],
                "actions": ["Define primary key", "Review outliers", "Add domain-specific checks"],
                "questions": ["What is the target variable?", "What’s a valid range for key metrics?"]
            },
            "text": None,
            "error": None
        }
    except Exception as e:
        return {"ok": False, "json": None, "text": None, "error": str(e)}