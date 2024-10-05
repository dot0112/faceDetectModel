def cal_iou(x: int, y: int, label: list[float], window_size: int) -> float:
    l_x1, l_y1, l_width, l_height = label
    l_x2, l_y2 = l_x1 + l_width, l_y1 + l_height
    w_x2, w_y2 = x + window_size, y + window_size

    if l_x2 <= x or w_x2 <= l_x1 or l_y2 <= y or w_y2 <= l_y1:
        return 0.0

    inter_x1 = max(l_x1, x)
    inter_y1 = max(l_y1, y)
    inter_x2 = min(l_x2, w_x2)
    inter_y2 = min(l_y2, w_y2)

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    label_area = l_width * l_height
    window_area = window_size * window_size
    union_area = label_area + window_area - inter_area

    return inter_area / union_area
