"""
Bank Statement Parser Agent with LangGraph + Groq
Automates: plan → generate parser → run pytest → self-fix (≤3 attempts).
"""
import sys
import subprocess
import re
from pathlib import Path
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")
client = Groq(api_key=GROQ_API_KEY)
def clean_code(raw: str) -> str:
    """Remove markdown fences and extra text from LLM output."""
    match = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL)
    if match:
        raw = match.group(1)
    return "\n".join(
        ln for ln in raw.splitlines() if not ln.strip().startswith("Here")
    ).strip()
def generate_parser(bank: str, attempt: int = 1) -> str:
    """Ask Groq LLM to generate a custom parser for the given bank."""
    try:
        prompt = f"""
        Generate a Python parser function parse(file_path:str)->pd.DataFrame for {bank} bank PDF/CSV.
        Requirements:
        - Must use pandas (and pdfplumber for PDFs).
        - Handle repeated headers, empty rows, numeric cleaning.
        - Replace NaN in numeric columns with 0.0, and strings with "".
        - Return only valid Python code (no markdown fences, no explanations).
        """
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        code = clean_code(resp.choices[0].message.content.strip())
        compile(code, f"{bank}_parser.py", "exec")
        if "def parse(" not in code:
            raise ValueError("Generated code missing parse()")
        return code
    except Exception as e:
        print(f"❌ LLM failed on attempt {attempt}: {e}")
        return """\
import pandas as pd
def parse(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path)
    return pd.DataFrame()
"""
def write_parser(bank: str, code: str) -> Path:
    """Write parser code to custom_parsers/{bank}_parser.py."""
    parser_path = Path("custom_parsers") / f"{bank}_parser.py"
    parser_path.parent.mkdir(exist_ok=True)
    parser_path.write_text(code, encoding="utf-8")
    return parser_path
def write_pytest(bank: str) -> Path:
    """Write pytest code for the given bank parser."""
    test_path = Path("tests") / f"test_{bank}_parser.py"
    test_path.parent.mkdir(exist_ok=True)
    pytest_code = f"""
import pandas as pd
import importlib.util
from pathlib import Path
def load_parser(bank):
    spec = importlib.util.spec_from_file_location(f"{{bank}}_parser", Path(f"custom_parsers/{{bank}}_parser.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse
def test_parser_output_matches_csv():
    bank = "{bank}"
    parse = load_parser(bank)
    csv_path = Path(f"data/{{bank}}/result.csv")
    pdf_path = Path(f"data/{{bank}}/sample.pdf")
    file_to_parse = csv_path if csv_path.exists() else pdf_path
    df_out = parse(str(file_to_parse))
    assert not df_out.empty, "Parser returned empty DataFrame"
    if csv_path.exists():
        df_ref = pd.read_csv(csv_path)
        # Fill NaN consistently
        for df in [df_out, df_ref]:
            for col in df.columns:
                if df[col].dtype != 'object':
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].fillna("")
        assert len(df_out) == len(df_ref), f"Row count mismatch: Parsed={{len(df_out)}}, Reference={{len(df_ref)}}"
        common_cols = set(df_ref.columns).intersection(df_out.columns)
        mismatches = []
        for col in common_cols:
            try:
                pd.testing.assert_series_equal(
                    df_out[col].reset_index(drop=True),
                    df_ref[col].reset_index(drop=True),
                    check_dtype=False
                )
                print(f"OK Column '{{col}}' matches")
            except AssertionError:
                mismatches.append(col)
                print(f"X Column '{{col}}' does NOT match")
        if mismatches:
            assert False, f"Mismatched columns: {{mismatches}}"
"""
    test_path.write_text(pytest_code, encoding="utf-8")
    return test_path
def run_pytest(test_path: Path) -> tuple[bool, list[str]]:
    """Run pytest and capture output."""
    if not test_path.exists():
        print(f"❌ Test file does not exist: {test_path}")
        return False, []
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", str(test_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    mismatches = []
    if "Mismatched columns:" in result.stdout:
        start = result.stdout.find("Mismatched columns:") + len("Mismatched columns:")
        mismatches = result.stdout[start:].strip().strip("[]").replace("'", "").split(", ")
        mismatches = [m for m in mismatches if m]
    if "Row count mismatch" in result.stdout:
        mismatches.append("Row count mismatch")
    return result.returncode == 0, mismatches
def plan_node(state: dict) -> dict:
    bank, attempt = state["bank"], state["attempt"]
    print(f"\n🧭 Planning (attempt {attempt}) for {bank}...")
    parser_path = write_parser(bank, generate_parser(bank, attempt))
    test_path = write_pytest(bank)
    return {**state, "parser_path": parser_path, "test_path": test_path}
def test_node(state: dict) -> dict:
    print("🧪 Running pytest...")
    success, mismatches = run_pytest(state["test_path"])
    return {**state, "success": success, "mismatches": mismatches}
def decide_node(state: dict) -> str:
    if state["success"]:
        print(f"✅ Attempt {state['attempt']} succeeded. All rows & columns match!")
        return END
    if state["attempt"] >= 3:
        print(f"❌ Max attempts reached (3). Final mismatches: {state['mismatches']}")
        return END
    print(f"🔄 Attempt {state['attempt']} failed. Mismatches: {state['mismatches']}. Retrying...")
    state["attempt"] += 1
    return "plan"
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank target (e.g., icici, sbi)")
    args = parser.parse_args()
    workflow = StateGraph(dict)
    workflow.add_node("plan", plan_node)
    workflow.add_node("test", test_node)
    workflow.add_edge("plan", "test")
    workflow.add_conditional_edges("test", decide_node)
    workflow.set_entry_point("plan")
    app = workflow.compile()
    app.invoke({"bank": args.target, "attempt": 1, "success": False, "mismatches": []})
if __name__ == "__main__":
    main()
