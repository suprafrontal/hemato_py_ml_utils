from .intersection_over_union import bounding_box_intersection_over_union


def test_bounding_box_intersection_over_union():
    r1 = [0.0, 0.0, 1.0, 1.0]
    r2 = [0.0, 0.0, 1.0, 1.0]
    delta = bounding_box_intersection_over_union(r1, r2)
    assert delta == 1.0
