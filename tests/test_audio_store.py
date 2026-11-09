"""Unit tests for the TTL audio store."""

from __future__ import annotations

import os
import time

from app.audio.store import AudioStore


def test_save_and_retrieve(tmp_path):
    store = AudioStore(tmp_path, ttl_seconds=3600)
    audio_id = store.save(b"ID3audio")
    path = store.path_for(audio_id)
    assert path is not None and path.read_bytes() == b"ID3audio"


def test_retrieval_survives_multiple_reads(tmp_path):
    store = AudioStore(tmp_path, ttl_seconds=3600)
    audio_id = store.save(b"abc")
    assert store.path_for(audio_id) is not None
    assert store.path_for(audio_id) is not None  # not deleted on first read


def test_rejects_path_traversal_and_malformed_ids(tmp_path):
    store = AudioStore(tmp_path, ttl_seconds=3600)
    assert store.path_for("../../etc/passwd") is None
    assert store.path_for("not-a-uuid") is None
    assert store.path_for("") is None


def test_expired_file_is_unavailable_and_removed(tmp_path):
    store = AudioStore(tmp_path, ttl_seconds=1)
    audio_id = store.save(b"data")
    f = tmp_path / f"{audio_id}.mp3"
    old = time.time() - 10
    os.utime(f, (old, old))
    store._last_sweep = 0  # force a sweep
    assert store.path_for(audio_id) is None
    assert not f.exists()


def test_purge_all(tmp_path):
    store = AudioStore(tmp_path, ttl_seconds=3600)
    store.save(b"a")
    store.save(b"b")
    assert store.purge_all() == 2
    assert list(tmp_path.glob("*.mp3")) == []
