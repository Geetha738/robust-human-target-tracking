| Scenario | Method | Accuracy | Frames locked on WRONG person | Frames to re-lock after he is visible |
|---|---|---|---|---|
| ByteTrack keeps same ID (easy) | Existing (YOLO+ByteTrack) | 97% | 0 | 0 |
| ByteTrack keeps same ID (easy) | + our reacquisition | 97% | 0 | 1 |
| ByteTrack keeps same ID (easy) | + occlusion reasoning (ours) | 97% | 0 | 1 |
| New ID after hiding behind obstacle | Existing (YOLO+ByteTrack) | 42% | 0 | NEVER |
| New ID after hiding behind obstacle | + our reacquisition | 96% | 0 | 2 |
| New ID after hiding behind obstacle | + occlusion reasoning (ours) | 96% | 0 | 2 |
| Hidden behind another person | Existing (YOLO+ByteTrack) | 53% | 0 | NEVER |
| Hidden behind another person | + our reacquisition | 94% | 0 | 2 |
| Hidden behind another person | + occlusion reasoning (ours) | 94% | 0 | 2 |
| Leaves frame, look-alike appears first | Existing (YOLO+ByteTrack) | 38% | 0 | NEVER |
| Leaves frame, look-alike appears first | + our reacquisition | 38% | 118 | NEVER |
| Leaves frame, look-alike appears first | + occlusion reasoning (ours) | 95% | 0 | 2 |
| Stranger stands where he vanished | Existing (YOLO+ByteTrack) | 41% | 0 | NEVER |
| Stranger stands where he vanished | + our reacquisition | 96% | 0 | 2 |
| Stranger stands where he vanished | + occlusion reasoning (ours) | 96% | 0 | 2 |
| People cross, IDs swap | Existing (YOLO+ByteTrack) | 49% | 48 | n/a |
| People cross, IDs swap | + our reacquisition | 92% | 3 | n/a |
| People cross, IDs swap | + occlusion reasoning (ours) | 92% | 3 | n/a |
| Hidden for a long time (6 s) | Existing (YOLO+ByteTrack) | 34% | 0 | NEVER |
| Hidden for a long time (6 s) | + our reacquisition | 93% | 0 | 5 |
| Hidden for a long time (6 s) | + occlusion reasoning (ours) | 93% | 0 | 5 |
| Blocker takes the ID (similar outfit) | Existing (YOLO+ByteTrack) | 38% | 78 | NEVER |
| Blocker takes the ID (similar outfit) | + our reacquisition | 94% | 3 | 7 |
| Blocker takes the ID (similar outfit) | + occlusion reasoning (ours) | 94% | 3 | 7 |
| Blocker takes the ID (identical outfit) | Existing (YOLO+ByteTrack) | 38% | 78 | NEVER |
| Blocker takes the ID (identical outfit) | + our reacquisition | 90% | 5 | 12 |
| Blocker takes the ID (identical outfit) | + occlusion reasoning (ours) | 90% | 5 | 12 |
| Target passes behind standing person | Existing (YOLO+ByteTrack) | 35% | 92 | NEVER |
| Target passes behind standing person | + our reacquisition | 96% | 1 | 2 |
| Target passes behind standing person | + occlusion reasoning (ours) | 96% | 1 | 2 |

**Average accuracy:** Existing (YOLO+ByteTrack) 47%, + our reacquisition 89%, + occlusion reasoning (ours) 94%