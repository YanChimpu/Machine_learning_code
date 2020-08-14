def single_nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sorted.argsort(scores)
    order = order[::-1]

    keep = []

    while order.size > 0:
        if order.size == 1:
            keep.append(order[0])
            break
        else:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[1:])
            yy1 = np.maximum(y1[i], y1[1:])
            xx2 = np.minimum(x2[i], x2[1:])
            yy2 = np.minimum(y2[i], y2[1:])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inner_area = w * h
            IOU = inner_area / (areas[i] + areas[order[1:]] - inner_area)
            inds = np.where(iou < threshold)[0]
            if inds.size == 0:
                break
            order = order[inds+1]
    return keep
