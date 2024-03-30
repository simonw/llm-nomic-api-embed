import llm
import pytest


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}


@pytest.mark.vcr()
def test_embed():
    model = llm.get_embedding_model("nomic-embed-text-v1.5-64")
    result = model.embed("hello world")
    assert isinstance(result, list)
    assert all(isinstance(item, float) for item in result)
    assert len(result) == 64
