from llm_video_editor.schemas import SceneCut
from llm_video_editor.stages.broll import _BM25, _Doc, filename_tags, tokenize


def test_filename_tags_drop_camera_codes():
    assert filename_tags("drone/DJI_0042_city-skyline_sunset.MP4") == [
        "drone", "city", "skyline", "sunset"]
    assert filename_tags("C0042.MP4") == []
    assert filename_tags("coffeePourCloseUp_v2.mov") == ["coffee", "pour", "close"]


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("We are pouring the coffees into cups") == ["pour", "coffe", "cup"]


def test_bm25_ranks_relevant_doc_first():
    docs = [_Doc("a", "city skyline drone sunset", 5, "landscape", [SceneCut(start=0, end=5)], None),
            _Doc("b", "pouring coffee closeup morning", 5, "portrait", [], None),
            _Doc("c", "laptop typing code desk", 5, "landscape", [], None)]
    bm = _BM25(docs)
    q = tokenize("every morning starts with fresh coffee")
    assert max(docs, key=lambda d: bm.score(q, d)).item_id == "b"
    assert bm.score(tokenize("underwater shark"), docs[0]) == 0
