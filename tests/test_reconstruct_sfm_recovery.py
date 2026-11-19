"""Weak SIFT recovery preserves native identities and historical observations."""

import json
from pathlib import Path
import sqlite3

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pycolmap")
from lingshu.world.reconstruct import colmap


def _database(path, counts=(100, 800)):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, params BLOB);
            CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT UNIQUE, camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, type INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER);
            CREATE TABLE "rig metadata"(rig_id INTEGER PRIMARY KEY, value TEXT);
            CREATE UNIQUE INDEX "rig value" ON "rig metadata"(value);
        ''')
        db.execute("INSERT INTO cameras VALUES(7, ?)", (b"unchanged intrinsics",))
        db.execute('INSERT INTO "rig metadata" VALUES(3, ?)', ("retained rig",))
        for image_id, count in enumerate(counts, start=1):
            db.execute("INSERT INTO images VALUES(?,?,7)", (image_id, f"view-{image_id}.jpg"))
            _feature(db, image_id, count, b"original")


def _feature(db, image_id, count, payload):
    db.execute("INSERT OR REPLACE INTO keypoints VALUES(?,?,6,?)", (image_id, count, payload + b" keypoints"))
    db.execute("INSERT OR REPLACE INTO descriptors VALUES(?,5,?,128,?)", (image_id, count, payload + b" descriptors"))


def _rows(path, table):
    with sqlite3.connect(path) as db:
        return db.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()


def _recover(tmp_path, database, preexisting=False):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return colmap._recover_weak_sift(
        tmp_path / "colmap", database, logs, ["--database_path", str(database),
                                              "--image_path", str(tmp_path / "images"),
                                              "--FeatureExtraction.use_gpu", "0",
                                              "--FeatureExtraction.num_threads", "2"],
        database_preexisting=preexisting,
    )


def test_only_weak_images_are_recovered_from_metadata_only_scratch(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    _database(database)
    original_strong = _rows(database, "keypoints")[1]
    original_camera, original_images = _rows(database, "cameras"), _rows(database, "images")
    calls = []
    def native(executable, command, args, log, cwd):
        calls.append(command)
        scratch = Path(args[args.index("--database_path") + 1])
        assert scratch != database
        assert _rows(scratch, "keypoints") == _rows(scratch, "descriptors") == []
        assert _rows(scratch, "cameras") == original_camera
        assert _rows(scratch, "images") == original_images
        assert _rows(scratch, "rig metadata") == [(3, "retained rig")]
        image_list = Path(args[args.index("--image_list_path") + 1])
        assert image_list.read_text(encoding="utf8").splitlines() == ["view-1.jpg"]
        assert args[args.index("--SiftExtraction.peak_threshold") + 1] == "0.002"
        assert args[args.index("--FeatureExtraction.num_threads") + 1] == "2"
        with sqlite3.connect(scratch) as db:
            _feature(db, 1, 720, b"recovered")
    monkeypatch.setattr(colmap, "run_colmap", native)
    result = _recover(tmp_path, database)
    assert calls == ["feature_extractor"]
    assert result["status"] == "recovered"
    assert result["minimum_features"] == 512
    assert result["retried_images"] == ["view-1.jpg"]
    assert result["initial_counts"] == {"view-1.jpg": 100, "view-2.jpg": 800}
    assert result["recovered_counts"] == {"view-1.jpg": 720}
    assert _rows(database, "keypoints")[1] == original_strong
    assert _rows(database, "descriptors")[0][1:4] == (5, 720, 128)
    assert _rows(database, "cameras") == original_camera
    assert _rows(database, "images") == original_images


def test_supported_images_are_not_reextracted(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    _database(database, (512, 800))
    before = database.read_bytes()
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: pytest.fail("unnecessary recovery"))
    result = _recover(tmp_path, database)
    assert result["status"] == "not_needed"
    assert result["retried_images"] == []
    assert database.read_bytes() == before


def test_preexisting_database_and_historical_matches_are_preserved(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    _database(database)
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO matches VALUES(1,42)")
    before = database.read_bytes()
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: pytest.fail("historical database recovery"))
    result = _recover(tmp_path, database, preexisting=True)
    assert result["status"] == "skipped_preexisting_database"
    assert database.read_bytes() == before


def test_new_database_with_matches_fails_before_feature_replacement(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    _database(database)
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO two_view_geometries VALUES(1,0)")
    before = database.read_bytes()
    monkeypatch.setattr(colmap, "run_colmap", lambda *args: pytest.fail("matched database recovery"))
    with pytest.raises(RuntimeError, match="match"):
        _recover(tmp_path, database)
    assert database.read_bytes() == before


def test_failed_reextraction_leaves_original_features_intact(tmp_path, monkeypatch):
    database = tmp_path / "database.db"
    _database(database)
    before = database.read_bytes()
    def failed(*args):
        raise RuntimeError("native extraction failed")
    monkeypatch.setattr(colmap, "run_colmap", failed)
    with pytest.raises(RuntimeError, match="native extraction failed"):
        _recover(tmp_path, database)
    assert database.read_bytes() == before
    receipt = json.loads((tmp_path / "logs/sift_support.json").read_text(encoding="utf8"))
    assert receipt["status"] == "failed"
    assert receipt["initial_counts"]["view-1.jpg"] == 100


@pytest.mark.parametrize("mutation", ["matches", "features", "camera"])
def test_concurrent_or_identity_changes_fail_closed(tmp_path, monkeypatch, mutation):
    database = tmp_path / "database.db"
    _database(database)
    before_feature = _rows(database, "keypoints")[0]
    def native(executable, command, args, log, cwd):
        scratch = Path(args[args.index("--database_path") + 1])
        with sqlite3.connect(scratch) as db:
            _feature(db, 1, 720, b"recovered")
            if mutation == "camera":
                db.execute("UPDATE cameras SET params=?", (b"changed",))
        if mutation != "camera":
            with sqlite3.connect(database) as db:
                if mutation == "matches":
                    db.execute("INSERT INTO matches VALUES(1,42)")
                else:
                    _feature(db, 1, 101, b"concurrent")
    monkeypatch.setattr(colmap, "run_colmap", native)
    with pytest.raises(RuntimeError, match="changed|match|identity"):
        _recover(tmp_path, database)
    if mutation != "features":
        assert _rows(database, "keypoints")[0] == before_feature
    else:
        assert _rows(database, "keypoints")[0][1] == 101
