from __future__ import annotations

from simple.envs.video_writer import VideoWriter


class _StubCvWriter:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _writer_without_ffmpeg(path) -> tuple[VideoWriter, _StubCvWriter]:
    path.write_bytes(b"recorded-video")
    cv_writer = _StubCvWriter()
    writer = VideoWriter.__new__(VideoWriter)
    writer.video_writer = cv_writer
    writer.filename = str(path)
    writer.is_ffmpeg_installed = False
    return writer, cv_writer


def test_release_labels_success_without_ffmpeg(tmp_path) -> None:
    writer, cv_writer = _writer_without_ffmpeg(tmp_path / "head_stereo_left.mp4")

    final_path = writer.release(success=True)

    assert cv_writer.released
    assert final_path.endswith("head_stereo_left_success.mp4")
    assert not (tmp_path / "head_stereo_left.mp4").exists()
    assert (tmp_path / "head_stereo_left_success.mp4").read_bytes() == b"recorded-video"


def test_release_labels_failed_without_ffmpeg(tmp_path) -> None:
    writer, cv_writer = _writer_without_ffmpeg(tmp_path / "head_stereo_right.mp4")

    final_path = writer.release(success=False)

    assert cv_writer.released
    assert final_path.endswith("head_stereo_right_failed.mp4")
    assert not (tmp_path / "head_stereo_right.mp4").exists()
    assert (tmp_path / "head_stereo_right_failed.mp4").read_bytes() == b"recorded-video"


def test_release_is_idempotent(tmp_path) -> None:
    writer, _ = _writer_without_ffmpeg(tmp_path / "head_stereo_left.mp4")

    first_path = writer.release(success=False)
    second_path = writer.release(success=True)

    assert second_path == first_path
    assert (tmp_path / "head_stereo_left_failed.mp4").exists()
    assert not (tmp_path / "head_stereo_left_failed_success.mp4").exists()
