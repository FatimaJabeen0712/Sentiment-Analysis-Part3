import numpy as np
import pandas as pd
import pytest

from src.audience.bridge import guess_content_type, to_tables
from src.audience.language import language_type
from src.audience.pipeline import exports, run_audience_analysis
from src.audience.simulate import simulate_posts_and_comments
from src.audience.topics import classify
from src.schemas import MissingColumnError, PipelineError


@pytest.fixture(scope="module")
def sim():
    return simulate_posts_and_comments(n_posts=600, seed=3)


@pytest.fixture(scope="module")
def result(sim):
    return run_audience_analysis(*sim)


@pytest.mark.parametrize("text,topic", [
    ("Price please?", "price_inquiry"),
    ("Delivery kitne din mein hogi?", "delivery"),
    ("Any discount on this suit?", "discount"),
    ("Is this available in Karachi?", "availability"),
    ("Ordered last week, still waiting.", "order_status"),
    ("Inbox me details send kar dein.", "purchase_interest"),
])
def test_assignment_example_comments_get_the_right_topic(text, topic):
    assert classify(text)["topic"] == topic


def test_language_types():
    assert language_type("Very beautiful dress!") == "English"
    assert language_type("Price bohat zyada hai") == "Roman Urdu"
    assert language_type("بہت خوبصورت") == "Urdu"
    assert language_type("❤❤❤") == "Emoji / other"


def test_simulation_is_reproducible_and_labeled_simulated(sim):
    p2, c2 = simulate_posts_and_comments(n_posts=600, seed=3)
    pd.testing.assert_frame_equal(sim[0], p2)
    assert (sim[0].data_origin == "simulated").all() and (sim[1].data_origin == "simulated").all()


def test_pipeline_funnel_flags_simulated_and_recovers_planted_truth(result):
    f = result.funnel
    assert f["comments_in"] - f["empty_removed"] - f["spam_removed"] - f["orphan_removed"] - f["duplicate_removed"] == f["comments_analyzed"]
    assert result.contains_simulated and any("SIMULATED" in w for w in result.warnings)
    assert result.evaluation["topic_agreement"] > 0.85
    assert result.evaluation["anomaly_recall"] >= 0.5


def test_segments_cover_all_comments_and_are_named(result):
    assert result.segments["comments"].sum() == len(result.comments)
    assert result.segments["segment"].notna().all()
    assert 6 <= len(result.segments) <= 14


def test_recommendations_have_evidence_confidence_and_hypothesis_labels(result):
    r = result.recommendations
    assert len(r) > 3
    assert set(r.confidence) <= {"high", "medium", "low"}
    assert set(r.type) <= {"Finding", "Hypothesis to test"}
    assert r.recommendation.str.len().min() > 20


def test_exports_leave_out_ground_truth_and_identity_columns(result):
    for name, frame in exports(result).items():
        assert not [c for c in frame.columns if c.startswith("_")], name
        assert "author" not in " ".join(frame.columns).lower(), name


def test_validation_errors(sim):
    posts, comments = sim
    with pytest.raises(MissingColumnError):
        run_audience_analysis(posts.drop(columns=["likes"]), comments)
    with pytest.raises(MissingColumnError):
        run_audience_analysis(posts, comments.drop(columns=["comment_text"]))
    with pytest.raises(PipelineError):
        run_audience_analysis(pd.concat([posts, posts.head(1)]), comments)


def test_same_text_on_different_posts_is_kept_but_same_post_duplicates_are_removed():
    posts = pd.DataFrame({"post_id": ["a", "b"], "platform": "Instagram", "content_type": "Reel",
                          "likes": [10, 10], "comments_count": [3, 3]})
    comments = pd.DataFrame({"post_id": ["a", "a", "b", "zzz"], "comment_text": ["Price please?", "price please", "Price please?", "Price?"]})
    r = run_audience_analysis(posts, comments)
    assert r.funnel["duplicate_removed"] == 1 and r.funnel["orphan_removed"] == 1
    assert r.funnel["comments_analyzed"] == 2


def test_bridge_from_part1_csv_uses_only_real_numbers():
    raw = pd.DataFrame({"comment_id": ["1", "2", "3"], "text": ["Nice", "Price?", "Love it"],
                        "video_id": ["v1", "v1", "v2"], "video_title": ["Chanel review #shorts", "Chanel review #shorts", "Haul"]})
    posts, comments = to_tables(raw, brand="Chanel")
    assert set(posts.data_origin) == {"observed_youtube"}
    assert posts["likes"].isna().all() and posts["views"].isna().all()   # never invented
    assert posts.set_index("post_id").loc["v1", "comments_count"] == 2
    assert len(comments) == 3
    with pytest.raises(ValueError):
        to_tables(pd.DataFrame({"x": [1]}))


def test_content_type_guess_ignores_hashtag_stuffing():
    assert guess_content_type("Chanel bag #style #fashion") == "Other video"
    assert guess_content_type("My first Chanel unboxing") == "Unboxing / haul"
    assert guess_content_type("New perfume #shorts") == "Short video"
