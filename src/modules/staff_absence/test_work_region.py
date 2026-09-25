from ...core.roi import ROI
from .work_region import WorkRegion


def make_region(**kwargs):
    roi = ROI((0, 0, 100, 100))
    defaults = dict(
        region_id="R1",
        roi=roi,
        employee_name="An",
        employee_state="present",
        assign_threshold=3,
        away_threshold=3,
        missing_threshold=5,
    )
    defaults.update(kwargs)
    return WorkRegion(**defaults)


def person(track_id, hip):
    return {"track_id": track_id, "box": (0, 0, 10, 10), "hip": hip, "keypoints": []}


def test_unassigned_to_assigned():
    r = make_region()
    for _ in range(2):
        r.update([person("1", (50, 50))])
    assert r.state == WorkRegion.STATE_UNASSIGNED, "chưa đủ assign_threshold"

    r.update([person("1", (50, 50))])
    assert r.state == WorkRegion.STATE_ASSIGNED
    assert r.person_id == "1"
    print("OK: unassigned -> assigned")


def test_assigned_to_away_and_back():
    r = make_region()
    for _ in range(3):
        r.update([person("1", (50, 50))])
    assert r.state == WorkRegion.STATE_ASSIGNED

    for _ in range(3):
        r.update([person("1", (500, 500))])  # ra khỏi ROI nhưng vẫn detect trong frame
    assert r.state == WorkRegion.STATE_AWAY
    print("OK: assigned -> away")

    r.update([person("1", (50, 50))])
    assert r.state == WorkRegion.STATE_ASSIGNED
    print("OK: away -> assigned (quay lại ngay, đúng hành vi bản gốc)")


def test_bug_fix_away_eventually_resets_to_unassigned():
    """Đây là bug đã phát hiện trong main.py gốc: away không bao giờ
    reset về unassigned nếu người đó biến mất hẳn khỏi khung hình."""
    r = make_region()
    for _ in range(3):
        r.update([person("1", (50, 50))])
    assert r.state == WorkRegion.STATE_ASSIGNED

    for _ in range(3):
        r.update([person("1", (500, 500))])
    assert r.state == WorkRegion.STATE_AWAY

    # Track_id "1" biến mất HOÀN TOÀN khỏi detected_people (không chỉ ra
    # khỏi ROI, mà tracker cũng deregister luôn) đủ missing_threshold frame
    for _ in range(5):
        r.update([])
    assert r.state == WorkRegion.STATE_UNASSIGNED, "BUG chưa được vá!"
    assert r.person_id is None
    print("OK: away -> unassigned sau missing_threshold (bug đã vá)")


def test_wrong_person_does_not_hijack_region():
    """Người khác đi ngang qua ROI trong lúc chủ cũ đang away không được
    làm gián đoạn bộ đếm/định danh của chủ cũ."""
    r = make_region()
    for _ in range(3):
        r.update([person("1", (50, 50))])
    assert r.person_id == "1"

    for _ in range(3):
        r.update([person("1", (500, 500))])
    assert r.state == WorkRegion.STATE_AWAY

    # Người lạ "2" đi ngang qua ROI trong khi "1" vẫn còn trong khung hình
    r.update([person("1", (500, 500)), person("2", (50, 50))])
    assert r.person_id == "1", "không được đổi chủ region sang người lạ"
    assert r.state == WorkRegion.STATE_AWAY
    print("OK: người lạ đi ngang không hijack region của chủ cũ")


def test_employee_state_absence_locks_region():
    r = make_region(employee_state="absence")
    assert r.state == WorkRegion.STATE_ABSENCE

    for _ in range(5):
        r.update([person("1", (50, 50))])
    assert r.state == WorkRegion.STATE_ABSENCE, "absence phải khoá, không xử lý camera"

    r.set_employee_state("present")
    assert r.state == WorkRegion.STATE_UNASSIGNED
    print("OK: employee_state=absence khoá region, present mở khoá lại")


if __name__ == "__main__":
    test_unassigned_to_assigned()
    test_assigned_to_away_and_back()
    test_bug_fix_away_eventually_resets_to_unassigned()
    test_wrong_person_does_not_hijack_region()
    test_employee_state_absence_locks_region()
    print("\nTất cả test PASS.")
