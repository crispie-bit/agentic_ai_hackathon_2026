"""
§3 · 02 — Tool definition.

    uv run section_3_bedrock/02_tool_definition.py

A tool is a JSON Schema plus a sentence. The model reads the name, the
description and the input_schema, and decides whether to ask you to run it.
Nothing else about your function reaches it.
"""

import json
import urllib.error
import urllib.request

import _bootstrap  # noqa: F401

from _common import MODEL_ID, bedrock_runtime

# ---------------------------------------------------------------- the tool

GET_WEATHER = {
    "name": "get_weather",
    "description": (
        "Get the current temperature and wind speed at a location, live. "
        "Use this whenever you are asked about current weather anywhere. "
        "You must supply the latitude and longitude yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "decimal degrees, e.g. 1.29"},
            "longitude": {"type": "number", "description": "decimal degrees, e.g. 103.85"},
        },
        "required": ["latitude", "longitude"],
    },
}


def get_weather(latitude: float, longitude: float) -> str:
    """Returns the current weather for a given latitude and longitude."""
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={latitude}&longitude={longitude}"
           "&current=temperature_2m,wind_speed_10m")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"ERROR: could not reach the weather service ({exc})."

    now = data["current"]
    return (f"At {now['time']}: {now['temperature_2m']}"
            f"{data['current_units']['temperature_2m']}, wind "
            f"{now['wind_speed_10m']}{data['current_units']['wind_speed_10m']}.")


# ---------------------------------------------------------------- binding

def ask(client, question: str) -> dict:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": question}],
        "tools": [GET_WEATHER],          # the only new key vs file 01
        "max_tokens": 400,
    })
    return json.loads(
        client.invoke_model(modelId=MODEL_ID, body=body)["body"].read())


def show(envelope: dict) -> None:
    print(f"  stop_reason: {envelope['stop_reason']}")
    for block in envelope["content"]:
        if block["type"] == "tool_use":
            print(f"  tool_use:    {block['name']}({block['input']})")
        elif block["type"] == "text":
            print(f"  text:        {block['text'].strip()[:100]}")


def main() -> None:
    client = bedrock_runtime()

    # The tool is ordinary code. No model involved.
    print("get_weather(1.29, 103.85)")
    print(f"  {get_weather(1.29, 103.85)}\n")

    # A relevant question: the model supplies the coordinates from its own
    # knowledge, then asks for the temperature it cannot know.
    print("what is the weather in Singapore?")
    show(ask(client, "What is the weather in Singapore right now?"))

    # An irrelevant question: no tool call, it just answers.
    print("\nwho is the president of the United States?")
    show(ask(client, "Who is the president of the United States?"))


if __name__ == "__main__":
    main()
