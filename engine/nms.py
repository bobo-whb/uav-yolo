import torch
import numpy as np


def calculate_area(box) -> float:
    """
    Args:
        box (List[int]): [x1, y1, x2, y2]
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Args:
        box1 (np.ndarray): np.array([x1, y1, x2, y2])
        box2 (np.ndarray): np.array([x1, y1, x2, y2])
    """
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    width_height = (right_bottom - left_top).clip(min=0)
    return width_height[0] * width_height[1]


def calculate_bbox_iou(box1, box2) -> float:
    """Returns the ratio of intersection area to the union"""
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = calculate_intersection_area(np.array(box1), np.array(box2))
    return intersect / (area1 + area2 - intersect)


def calculate_bbox_ios(box1, box2) -> float:
    """Returns the ratio of intersection area to the smaller box's area"""
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = calculate_intersection_area(np.array(box1), np.array(box2))
    smaller_area = np.minimum(area1, area2)
    return intersect / smaller_area


def has_match(box1, box2, match_type: str = "IOU", match_threshold: float = 0.5) -> bool:
    if match_type == "IOU":
        threshold_condition = calculate_bbox_iou(box1, box2) > match_threshold
    elif match_type == "IOS":
        threshold_condition = calculate_bbox_ios(box1, box2) > match_threshold
    else:
        raise ValueError("Unknown match type")
    return threshold_condition


def greedy_nmm(object_predictions_as_tensor: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,6].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    keep_to_merge_list = {}

    # we extract coordinates for every prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # we extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P according to their confidence scores
    order = scores.argsort()

    while len(order) > 0:
        # extract the index of the prediction with highest score, call this prediction S
        idx = order[-1]

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            keep_to_merge_list[idx.tolist()] = []
            break

        # select coordinates of BBoxes according to the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P with the prediction S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union
        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P with the prediction S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError("Unknown match metric")

        # keep the boxes with IoU/IoS less than match_threshold
        mask = match_metric_value < match_threshold
        matched_box_indices = order[~mask].flip(dims=(0,))
        unmatched_indices = order[mask]

        # update box pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        # create keep_ind to merge_ind_list mapping
        keep_to_merge_list[idx.tolist()] = matched_box_indices.tolist()

    return keep_to_merge_list


def merge_boxes(box1, box2):
    """
    Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    left_top = np.minimum(box1[:2], box2[:2])
    right_bottom = np.maximum(box1[2:], box2[2:])
    return list(np.concatenate((left_top, right_bottom)))


def merge_scores(score1, score2):
    """
    Merges two scores by averaging
    """
    return max(score1, score2)


def batched_greedy_nmm(object_predictions_as_tensor: torch.tensor, match_metric: str = "IOU",
                       match_threshold: float = 0.5):
    """
    Apply greedy version of non-maximum merging per category to avoid detecting
    too many overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores and category ids, Shape: [num_boxes,6].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = greedy_nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
            keep_to_merge_list[keep] = merge_list
    return keep_to_merge_list


def GreedyNMMPostprocess(object_predictions, match_threshold, match_metric):
    object_predictions_as_tensor = object_predictions.clone().detach()
    keep_to_merge_list = batched_greedy_nmm(
        object_predictions_as_tensor,
        match_threshold=match_threshold,
        match_metric=match_metric,
    )

    selected_object_predictions = []
    for keep_ind, merge_ind_list in keep_to_merge_list.items():
        merged_box = object_predictions_as_tensor[keep_ind, :4].tolist()
        merged_score = object_predictions_as_tensor[keep_ind, 4].item()
        for merge_ind in merge_ind_list:
            merged_box = merge_boxes(merged_box, object_predictions_as_tensor[merge_ind, :4].tolist())
            merged_score = merge_scores(merged_score, object_predictions_as_tensor[merge_ind, 4].item())
        merged_prediction = merged_box + [merged_score] + [object_predictions_as_tensor[keep_ind, 5].item()]
        selected_object_predictions.append(merged_prediction)

    return selected_object_predictions
