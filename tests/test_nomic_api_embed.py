import llm
import pytest
from vcr.request import Request as VcrRequest
from vcr.stubs import httpx_stubs


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xa6\x00\x00\x01\x1a"
    b"\x02\x03\x00\x00\x00\xe6\x99\xc4^\x00\x00\x00\tPLTE\xff\xff\xff"
    b"\x00\xff\x00\xfe\x01\x00\x12t\x01J\x00\x00\x00GIDATx\xda\xed\xd81\x11"
    b"\x000\x08\xc0\xc0.]\xea\xaf&Q\x89\x04V\xe0>\xf3+\xc8\x91Z\xf4\xa2\x08EQ\x14E"
    b"Q\x14EQ\x14EQ\xd4B\x91$I3\xbb\xbf\x08EQ\x14EQ\x14EQ\x14E\xd1\xa5"
    b"\xd4\x17\x91\xc6\x95\x05\x15\x0f\x9f\xc5\t\x9f\xa4\x00\x00\x00\x00IEND\xaeB`"
    b"\x82"
)


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


@pytest.mark.vcr()
def test_embed_vision():
    model = llm.get_embedding_model("nomic-embed-vision-v1.5")
    result = model.embed(TINY_PNG)
    assert isinstance(result, list)
    assert all(isinstance(item, float) for item in result)
    assert len(result) == 768


@pytest.mark.vcr()
def test_embed_combination():
    model = llm.get_embedding_model("nomic-embed-combined-v1.5")
    result = model.embed_batch(["text", TINY_PNG, "more text"])
    assert isinstance(result, list)
    assert len(result) == 3
    for vector in result:
        assert all(isinstance(item, float) for item in vector)


# Nasty monkey-patch to work around https://github.com/kevin1024/vcrpy/issues/656
def _make_vcr_request(httpx_request, **kwargs):
    try:
        body = httpx_request.read().decode("utf-8")
    except UnicodeDecodeError:
        body = httpx_request.read().decode("ISO-8859-1").encode("utf-8")

    uri = str(httpx_request.url)
    headers = dict(httpx_request.headers)
    return VcrRequest(httpx_request.method, uri, body, headers)


httpx_stubs._make_vcr_request = _make_vcr_request
