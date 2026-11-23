"""
CatchCam — Week 3: Space-Time Graph + Journey Reconstruction
================================================================
Builds a graph where:
  - Each NODE is one (camera_id, track_id) sighting -- i.e. "this person was
    seen on this camera during this time window".
  - Each EDGE is a cross-camera match from Week 2, weighted by appearance
    similarity, connecting a sighting on one camera to a sighting on another.

Then reconstructs a "journey" by walking the graph from a starting sighting,
always following the strongest plausible next edge -- giving you the
Camera 1 -> Camera 3 -> Camera 5 style timeline described in your proposal.

Usage:
    # Build the graph and save it
    python build_graph.py --observations outputs/cam1_observations.json outputs/cam2_observations.json outputs/cam3_observations.json --matches matches/cross_camera_matches.json --out_dir graph

    # Reconstruct a journey starting from a specific sighting
    python build_graph.py --observations outputs/cam1_observations.json outputs/cam2_observations.json outputs/cam3_observations.json --matches matches/cross_camera_matches.json --start_camera cam1 --start_track 719

Requires: networkx (already in your venv)
"""
import argparse
import json
from pathlib import Path

import networkx as nx


def node_id(camera_id: str, track_id: int) -> str:
    """Unique node identifier for a (camera, track) sighting."""
    return f"{camera_id}:{track_id}"


def build_graph(observation_paths, matches_path):
    G = nx.DiGraph()

    # --- Add one node per (camera, track) sighting ---
    for obs_path in observation_paths:
        with open(obs_path) as f:
            observations = json.load(f)

        by_track = {}
        for obs in observations:
            key = (obs["camera_id"], obs["track_id"])
            if key not in by_track:
                by_track[key] = {"first": obs, "last": obs}
            else:
                if obs["frame"] < by_track[key]["first"]["frame"]:
                    by_track[key]["first"] = obs
                if obs["frame"] > by_track[key]["last"]["frame"]:
                    by_track[key]["last"] = obs

        for (camera_id, track_id), info in by_track.items():
            G.add_node(
                node_id(camera_id, track_id),
                camera_id=camera_id,
                track_id=track_id,
                first_seen_timestamp=info["first"]["timestamp"],
                last_seen_timestamp=info["last"]["timestamp"],
                first_seen_frame=info["first"]["frame"],
                last_seen_frame=info["last"]["frame"],
            )

    # --- Add weighted edges from Week 2's cross-camera matches ---
    with open(matches_path) as f:
        matches = json.load(f)

    for m in matches:
        a = node_id(m["camera_a"], m["track_a"])
        b = node_id(m["camera_b"], m["track_b"])
        if a in G.nodes and b in G.nodes:
            # Direction: earlier sighting -> later sighting (a plausible
            # real-world transition, "seen here, then seen there")
            a_time = G.nodes[a]["last_seen_timestamp"]
            b_time = G.nodes[b]["first_seen_timestamp"]
            if a_time <= b_time:
                src, dst = a, b
            else:
                src, dst = b, a
            # Keep the strongest edge if multiple matches land on the same pair
            if not G.has_edge(src, dst) or G[src][dst]["similarity"] < m["similarity"]:
                G.add_edge(src, dst, similarity=m["similarity"])

    return G


def reconstruct_journey(G: nx.DiGraph, start_camera: str, start_track: int, max_hops: int = 10):
    """
    Walk forward from a starting sighting, always following the highest-
    similarity outgoing edge, until no more edges exist or max_hops is hit.
    """
    start = node_id(start_camera, start_track)
    if start not in G.nodes:
        print(f"[CatchCam] No such node: {start}")
        return []

    journey = [start]
    current = start
    visited = {start}

    for _ in range(max_hops):
        out_edges = [
            (dst, data["similarity"])
            for _, dst, data in G.out_edges(current, data=True)
            if dst not in visited
        ]
        if not out_edges:
            break
        out_edges.sort(key=lambda x: -x[1])
        next_node, _ = out_edges[0]
        journey.append(next_node)
        visited.add(next_node)
        current = next_node

    # Format as a readable journey
    result = []
    for node in journey:
        attrs = G.nodes[node]
        result.append({
            "camera_id": attrs["camera_id"],
            "track_id": attrs["track_id"],
            "first_seen_timestamp": attrs["first_seen_timestamp"],
            "last_seen_timestamp": attrs["last_seen_timestamp"],
        })
    return result


def reconstruct_journey_backward(G: nx.DiGraph, start_camera: str, start_track: int, max_hops: int = 10):
    """
    Mirror of reconstruct_journey, but walks BACKWARD through incoming edges
    instead of forward through outgoing ones. Answers "where did this object
    come from?" instead of "where did this object go?".
    """
    start = node_id(start_camera, start_track)
    if start not in G.nodes:
        print(f"[CatchCam] No such node: {start}")
        return []

    journey = [start]
    current = start
    visited = {start}

    for _ in range(max_hops):
        in_edges = [
            (src, data["similarity"])
            for src, _, data in G.in_edges(current, data=True)
            if src not in visited
        ]
        if not in_edges:
            break
        in_edges.sort(key=lambda x: -x[1])
        prev_node, _ = in_edges[0]
        journey.append(prev_node)
        visited.add(prev_node)
        current = prev_node

    # Reverse so the result reads chronologically (earliest first), same as
    # the forward version -- the caller shouldn't need to know which
    # direction we walked internally.
    journey.reverse()

    result = []
    for node in journey:
        attrs = G.nodes[node]
        result.append({
            "camera_id": attrs["camera_id"],
            "track_id": attrs["track_id"],
            "first_seen_timestamp": attrs["first_seen_timestamp"],
            "last_seen_timestamp": attrs["last_seen_timestamp"],
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", nargs="+", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--out_dir", default="graph")
    parser.add_argument("--start_camera", default=None,
                         help="If set, reconstruct a journey starting here")
    parser.add_argument("--start_track", type=int, default=None)
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward",
                         help="forward = 'where did it go?' (default), "
                              "backward = 'where did it come from?'")
    args = parser.parse_args()

    G = build_graph(args.observations, args.matches)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    graph_file = out_path / "space_time_graph.gml"
    # GML needs plain types -- cast anything odd to str/float first
    nx.write_gml(G, graph_file)

    print(f"[CatchCam] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[CatchCam] Saved -> {graph_file}")

    if args.start_camera is not None and args.start_track is not None:
        if args.direction == "forward":
            journey = reconstruct_journey(G, args.start_camera, args.start_track)
            label = "Journey (forward: where did it go?)"
        else:
            journey = reconstruct_journey_backward(G, args.start_camera, args.start_track)
            label = "Journey (backward: where did it come from?)"

        print(f"\n[CatchCam] {label} starting from {args.start_camera} track {args.start_track}:")
        for i, hop in enumerate(journey):
            arrow = "  ->  " if i > 0 else "      "
            print(f"{arrow}{hop['camera_id']} (track {hop['track_id']}), "
                  f"seen {hop['first_seen_timestamp']:.2f}s - {hop['last_seen_timestamp']:.2f}s")

        journey_file = out_path / f"journey_{args.direction}_{args.start_camera}_{args.start_track}.json"
        with open(journey_file, "w") as f:
            json.dump(journey, f, indent=2)
        print(f"\n[CatchCam] Journey saved -> {journey_file}")


if __name__ == "__main__":
    main()
