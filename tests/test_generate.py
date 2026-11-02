from musicgen.generate import extract_songs


def test_extract_songs_single():
    text = "junk\n\nX:1\nT:Test Tune\nK:C\nABC|\n\ntrailing garbage"
    songs = extract_songs(text)
    assert len(songs) == 1
    assert songs[0].startswith("X:1")


def test_extract_songs_multiple():
    text = "X:1\nT:A\nK:C\nAB|\n\nX:2\nT:B\nK:D\nCD|\n\nX:3\nT:C\nK:E\nEF|"
    songs = extract_songs(text)
    assert len(songs) == 3
    assert [s.splitlines()[0] for s in songs] == ["X:1", "X:2", "X:3"]


def test_extract_songs_none():
    assert extract_songs("no headers here at all") == []
