import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab"


@contextlib.contextmanager
def patched_modules(modules):
    old = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@contextlib.contextmanager
def lab_import_path(path: Path):
    old_path = sys.path[:]
    old_bootstrap = sys.modules.pop("_bootstrap", None)
    sys.path[:0] = [str(path.parent), str(LAB)]
    try:
        yield
    finally:
        sys.path[:] = old_path
        if old_bootstrap is None:
            sys.modules.pop("_bootstrap", None)
        else:
            sys.modules["_bootstrap"] = old_bootstrap


def load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with lab_import_path(path):
        spec.loader.exec_module(module)
    return module


class LabSmokeTests(unittest.TestCase):
    def test_foundations_memory_path_uses_previous_turns(self) -> None:
        common = types.ModuleType("_common")
        common.GROQ_MODEL = "fake-model"
        common.banner = lambda _title: None
        common.groq_client = lambda: None
        common.report_usage = lambda _label, _usage: None

        with patched_modules({"_common": common}):
            module = load_file(LAB / "section_1_foundation" / "01_foundations.py", "foundations_smoke")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            module.part_b_statelessness(FakeGroqClient())

        self.assertIn("Does call 2 know the name?  YES", out.getvalue())

    def test_agent_lab_runs_with_mocked_model_and_tools(self) -> None:
        modules = fake_langchain_modules()
        common = types.ModuleType("_common")
        common.banner = lambda title: print(f"== {title} ==")
        common.chat_model = lambda temperature=0: FakeAgentModel()
        modules["_common"] = common

        out = io.StringIO()
        with patched_modules(modules), contextlib.redirect_stdout(out):
            load_file(LAB / "section_2_agentic_ai_basic" / "05_agent_lab.py", "agent_lab_smoke")

        text = out.getvalue()
        self.assertIn("[tool] lookup_order('A-1042')", text)
        self.assertIn("[tool] track_shipment('Fleetline', 'FL77213')", text)
        self.assertIn("Ended: the model stopped asking for tools.", text)

    def test_rag_retrieval_stays_offline(self) -> None:
        common = types.ModuleType("_common")
        common.GROQ_MODEL = "fake-model"
        common.banner = lambda _title: None
        common.groq_client = lambda: None
        common.report_usage = lambda _label, _usage: None

        with patched_modules({"_common": common}):
            module = load_file(LAB / "section_2_agentic_ai_basic" / "06d_rag.py", "rag_smoke")

        index = module.build_index(module.DOCUMENTS)
        results = module.retrieve("How many points for a free drink?", index)

        self.assertGreater(len(results), 0)
        self.assertIn("One hundred points", results[0][1]["text"])


class FakeUsage:
    prompt_tokens = 1
    completion_tokens = 1
    total_tokens = 2


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeCompletions:
    def create(self, *, messages, **_kwargs):
        last = messages[-1]["content"]
        if "What is my name" in last:
            content = "Your name is Alex and you work on the payments team."
        else:
            content = "Noted."
        return FakeResponse(content)


class FakeChat:
    completions = FakeCompletions()


class FakeGroqClient:
    chat = FakeChat()


class FakeBaseMessage:
    def __init__(self, content=None, **kwargs):
        self.content = content or ""
        self.tool_call_id = kwargs.get("tool_call_id")


class FakeReply(FakeBaseMessage):
    def __init__(self, content="", tool_calls=()):
        super().__init__(content)
        self.tool_calls = list(tool_calls)


class FakeAgentModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return FakeReply(tool_calls=[{
                "name": "lookup_order",
                "args": {"order_id": "A-1042"},
                "id": "call-1",
            }])
        if self.calls == 2:
            return FakeReply(tool_calls=[{
                "name": "track_shipment",
                "args": {"carrier": "Fleetline", "tracking": "FL77213"},
                "id": "call-2",
            }])
        return FakeReply("REFUND. No scan for 6 days, past the 5-day threshold.")


class FakeTool:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def invoke(self, args):
        return self.func(**args)


def fake_tool(func):
    return FakeTool(func)


def fake_langchain_modules():
    messages = types.ModuleType("langchain_core.messages")
    messages.HumanMessage = FakeBaseMessage
    messages.SystemMessage = FakeBaseMessage
    messages.ToolMessage = FakeBaseMessage

    tools = types.ModuleType("langchain_core.tools")
    tools.tool = fake_tool

    package = types.ModuleType("langchain_core")
    return {
        "langchain_core": package,
        "langchain_core.messages": messages,
        "langchain_core.tools": tools,
    }


if __name__ == "__main__":
    unittest.main()
