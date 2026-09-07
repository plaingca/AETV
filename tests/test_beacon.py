import numpy as np
import pytest

from aetv import beacon, golay


def _mix_word(chips, word, difference, strength=0.55):
    start = beacon.SYNC_LEN + 24 * word
    original = golay.decode_soft(chips[start:start + 24])
    other = 1 - 2 * golay.codeword_bits(original ^ difference)
    chips[start:start + 24] = (1 - strength) * chips[start:start + 24] + strength * other


def _weak_pair(counter=1000, step=45, second_callsign="W9XYZ/3"):
    left = beacon.encode_superframe(counter, "W9XYZ/3", 2)
    right = beacon.encode_superframe((counter + step) & 1023, second_callsign, 2)
    # Different faded identity words can be repaired by the other observation.
    _mix_word(left, 1, 13, 0.70)
    _mix_word(right, 2, 19, 0.70)
    # Each frame also needs a secondary soft candidate in its own variable word.
    _mix_word(left, 0, 4)
    _mix_word(right, 5, 1)
    return left, right


def test_golay_batch_scores_match_individual_decoder():
    soft = np.random.default_rng(941).standard_normal((7, 24))
    scores = golay.soft_scores(soft)
    assert scores.shape == (7, 4096)
    assert np.array_equal(scores.argmax(axis=1), [golay.decode_soft(row) for row in soft])


def test_beacon_known_padding_repairs_a_wrong_final_word():
    chips = beacon.encode_superframe(209, "K4ABC", 8)
    _mix_word(chips, 6, 65)
    assert beacon.decode_superframe(chips[beacon.SYNC_LEN:]) == (209, "K4ABC", 8)


def test_beacon_known_mode_cannot_rescue_a_single_failed_crc():
    chips = beacon.encode_superframe(318, "VE3TEST", 8)
    _mix_word(chips, 5, (1 << 10) | (1 << 7))
    assert beacon.decode_superframe(chips[beacon.SYNC_LEN:]) is None
    assert beacon.decode_superframe(chips[beacon.SYNC_LEN:], expected_mode=8) is None
    assert beacon.decode_superframe(chips[beacon.SYNC_LEN:], expected_mode=7) is None
    # It remains a useful candidate when a second frame can corroborate it.
    assert (318, "VE3TEST", 8) in beacon._payload_candidates(chips[beacon.SYNC_LEN:], 8)


@pytest.mark.parametrize("step", [45, 46])
@pytest.mark.parametrize("gain", [1.0, -1e-4])
def test_weak_beacon_pair_requires_both_crcs_and_handles_counter_wrap(step, gain):
    left, right = _weak_pair(step=step)
    assert beacon.find_beacon_superframe(left, expected_mode=2) is None
    assert beacon.find_beacon_superframe(right, expected_mode=2) is None
    stream = gain * np.concatenate([np.zeros(7), left, right])
    found = beacon.find_beacon_superframe(stream, expected_mode=2)
    assert found is not None
    assert found.callsign == "W9XYZ/3" and found.mode_index == 2
    assert found.frame_index == (1000 + step) & 1023
    assert found.chip_offset == 7 + beacon.SUPERFRAME_LEN


@pytest.mark.parametrize("step,second_callsign", [(44, "W9XYZ/3"), (47, "W9XYZ/3"), (45, "K8OTHER")])
def test_weak_beacon_pair_rejects_counter_or_identity_disagreement(step, second_callsign):
    left, right = _weak_pair(step=step, second_callsign=second_callsign)
    assert beacon.find_beacon_superframe(np.concatenate([left, right]), expected_mode=2) is None


def test_beacon_scan_continues_past_a_stronger_wrong_mode():
    wrong = beacon.encode_superframe(10, "WRONG", 7)
    right = beacon.encode_superframe(100, "RIGHT", 8)
    right[:beacon.SYNC_LEN] += np.random.default_rng(521).normal(0, 0.2, beacon.SYNC_LEN)
    found = beacon.find_beacon_superframe(np.concatenate([wrong, right]), expected_mode=8)
    assert found is not None and found.callsign == "RIGHT"


def test_beacon_crc_lookup_matches_direct_check():
    table, affine = beacon._crc_syndromes()
    rng = np.random.default_rng(164)
    for _ in range(50):
        words = rng.integers(0, 4096, size=7)
        bits = ((words[:, None] >> np.arange(11, -1, -1)) & 1).reshape(-1)
        direct = beacon.crc16(bits[:62]) ^ beacon._bits_to_int(bits[62:78])
        assert (np.bitwise_xor.reduce(table[np.arange(7), words]) ^ affine) == direct


def test_beacon_rejects_noise_even_with_repeated_perfect_sync():
    rng = np.random.default_rng(217)
    for _ in range(64):
        stream = rng.standard_normal(3 * beacon.SUPERFRAME_LEN)
        for start in range(0, len(stream), beacon.SUPERFRAME_LEN):
            stream[start:start + beacon.SYNC_LEN] = beacon.SYNC
        assert beacon.find_beacon_superframe(stream, expected_mode=8) is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_beacon_rejects_nonfinite_chips(bad):
    chips = beacon.encode_superframe(0, "N0CALL", 8)
    chips[-1] = bad
    assert beacon.find_beacon_superframe(chips, expected_mode=8) is None
    assert beacon.decode_superframe(chips[beacon.SYNC_LEN:]) is None
