"""Frame-level identity qualification; reads observations and model priors only.

ACCEPT/DEFER/REJECT here qualify an association, not the truth or authority of
the observation. The existing radius is in model coordinates, not metres.
"""
from copy import deepcopy
import math


POLICIES = ("legacy", "conservative")
RADIUS = 2.0


def validate_policy(policy):
    if policy not in POLICIES:
        raise ValueError("identity_policy must be legacy or conservative")
    return policy


def validate_measurements(observations):
    """Validate conservative measurements without association or state changes."""
    if not isinstance(observations, (list, tuple)):
        raise ValueError("observations must be a list of measurements")
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("each observation must be a mapping")
        category, pos = observation.get("category"), observation.get("pos")
        if not isinstance(category, str) or not category:
            raise ValueError("category must be a nonempty string")
        if (not isinstance(pos, (list, tuple)) or len(pos) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in pos)):
            raise ValueError("pos must contain three finite numbers")
        if not isinstance(observation.get("conditions", {}), dict):
            raise ValueError("conditions must be a mapping")
        eid = observation.get("eid")
        if eid is not None and (isinstance(eid, bool) or not isinstance(eid, (str, int))):
            raise ValueError("eid must be a string, integer or null")


def inspect_frame(nodes, predictions, observations, target_tick):
    """Return detached receipts without allocating IDs or changing any state.

    All candidates are retained. Work and diagnostic space are O(observations
    * known entities); truncating the set could fabricate uniqueness.
    """
    validate_measurements(observations)
    rows = []
    for index, observation in enumerate(observations):
        eid = observation.get("eid")
        eid = "" if eid is None else str(eid)
        rows.append({"observation_index": index, "tick": target_tick,
                     "measurement": deepcopy(observation),
                     "explicit_eid": eid or None, "matched_eid": None,
                     "association_state": "ACCEPT", "reason": "new_track",
                     "candidates": []})
    explicit_counts = {}
    for row in rows:
        if row["explicit_eid"]:
            eid = row["explicit_eid"]
            explicit_counts[eid] = explicit_counts.get(eid, 0) + 1
    claims = {}
    for row in rows:
        eid = row["explicit_eid"]
        if eid:
            if explicit_counts[eid] > 1:
                row.update(association_state="REJECT", reason="duplicate_explicit_id")
            else:
                row.update(matched_eid=eid if eid in nodes else None,
                           reason="explicit_id")
            continue
        measurement = row["measurement"]
        for candidate, node in nodes.items():
            if (candidate in explicit_counts or node.category != measurement["category"]
                    or node.attrs.get("hypothesis")):
                continue
            prediction = predictions.get(candidate)
            fresh = (prediction is not None
                     and prediction.get("target_tick", target_tick) == target_tick)
            ref = prediction["predicted"] if fresh else node.pos
            distance = math.dist(ref, measurement["pos"])
            if distance < RADIUS:
                row["candidates"].append({"eid": candidate, "distance": distance,
                    "reference": list(ref),
                    "source": "prediction" if fresh else "last_observation"})
                claims[candidate] = claims.get(candidate, 0) + 1
        row["candidates"].sort(key=lambda c: (c["distance"], c["eid"]))
    for row in rows:
        if row["explicit_eid"] or not row["candidates"]:
            continue
        candidates = row["candidates"]
        if len(candidates) > 1:
            row.update(association_state="DEFER", reason="multiple_eligible_tracks")
        elif claims[candidates[0]["eid"]] > 1:
            row.update(association_state="DEFER", reason="competing_observations")
        else:
            row.update(matched_eid=candidates[0]["eid"], reason="unique_candidate")
    return {"tick": target_tick, "radius": RADIUS, "readings": rows,
            "accepted": sum(r["association_state"] == "ACCEPT" for r in rows),
            "deferred": sum(r["association_state"] == "DEFER" for r in rows),
            "rejected": sum(r["association_state"] == "REJECT" for r in rows)}


def validate_history(history, retained_entities):
    """Validate v2 association audit before a recovery candidate can be committed."""
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    for frame in history:
        if "identity_readings" not in frame:
            continue  # v2 may retain earlier legacy frames
        readings = frame["identity_readings"]
        if not isinstance(readings, list):
            raise ValueError("identity_readings must be a list")
        accepted = {}
        for index, row in enumerate(readings):
            if (row["observation_index"] != index or row["tick"] != frame["tick"]
                    or row["association_state"] not in ("ACCEPT", "DEFER", "REJECT")):
                raise ValueError("invalid identity receipt")
            inspect_frame({}, {}, [row["measurement"]], frame["tick"])
            if row["association_state"] == "ACCEPT":
                eid = row["resolved_eid"]
                if not isinstance(eid, str) or not eid or eid in accepted:
                    raise ValueError("accepted identity must have a unique resolved_eid")
                # #1011：与快照侧同一投影（world_model.perceive 落盘时逐值
                # float() 化）——旧实现拿原始 int 测量值与 float 快照做恒等比较，
                # >2^53 的合法整数坐标（validate_measurements 放行）使投影恒不等，
                # v2 恢复（wm_simloop seed_from_prior 的本函数门卫）永久失败。
                accepted[eid] = [float(v) for v in row["measurement"]["pos"]]
            elif row.get("resolved_eid") is not None:
                raise ValueError("unresolved measurement cannot have a resolved_eid")
        projected = {eid: pos for eid, pos in accepted.items() if eid in retained_entities}
        if projected != frame["entities"]:
            raise ValueError("identity receipts and confirmed observation history disagree")
