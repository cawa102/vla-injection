---
source_file: "tests/evasion_tax/metric/fixtures/libero_obs_spatial0.json"
type: "code"
community: "LIBERO Obs Fixture (Bowls)"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/LIBERO_Obs_Fixture_Bowls
---

# libero_obs_spatial0.json

## Connections
- [[LiberoStateAdapter]] - `shares_data_with` [INFERRED]
- [[akita_black_bowl_1_pos_1]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_1_quat_1]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_1_to_robot0_eef_pos_1]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_1_to_robot0_eef_quat_1]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_2_pos]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_2_quat]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_2_to_robot0_eef_pos]] - `defined_in` [EXTRACTED]
- [[akita_black_bowl_2_to_robot0_eef_quat]] - `defined_in` [EXTRACTED]
- [[cookies_1_pos]] - `defined_in` [EXTRACTED]
- [[cookies_1_quat]] - `defined_in` [EXTRACTED]
- [[cookies_1_to_robot0_eef_pos]] - `defined_in` [EXTRACTED]
- [[cookies_1_to_robot0_eef_quat]] - `defined_in` [EXTRACTED]
- [[domain_name_1]] - `defined_in` [EXTRACTED]
- [[glazed_rim_porcelain_ramekin_1_pos]] - `defined_in` [EXTRACTED]
- [[glazed_rim_porcelain_ramekin_1_quat]] - `defined_in` [EXTRACTED]
- [[glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos]] - `defined_in` [EXTRACTED]
- [[glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat]] - `defined_in` [EXTRACTED]
- [[language_instruction_3]] - `contains` [EXTRACTED]
- [[language_instruction_2]] - `defined_in` [EXTRACTED]
- [[libero_obs_goal_opendrawer.json]] - `conceptually_related_to` [INFERRED]
- [[obj_of_interest_1]] - `contains` [EXTRACTED]
- [[object-state_1]] - `defined_in` [EXTRACTED]
- [[obs_1]] - `contains` [EXTRACTED]
- [[plate_1_pos_1]] - `defined_in` [EXTRACTED]
- [[plate_1_quat_1]] - `defined_in` [EXTRACTED]
- [[plate_1_to_robot0_eef_pos_1]] - `defined_in` [EXTRACTED]
- [[plate_1_to_robot0_eef_quat_1]] - `defined_in` [EXTRACTED]
- [[problem_info_1]] - `contains` [EXTRACTED]
- [[problem_name_1]] - `defined_in` [EXTRACTED]
- [[robot0_eef_pos_1]] - `defined_in` [EXTRACTED]
- [[robot0_eef_quat_1]] - `defined_in` [EXTRACTED]
- [[robot0_gripper_qpos_1]] - `defined_in` [EXTRACTED]
- [[robot0_gripper_qvel_1]] - `defined_in` [EXTRACTED]
- [[robot0_joint_pos_1]] - `defined_in` [EXTRACTED]
- [[robot0_joint_pos_cos_1]] - `defined_in` [EXTRACTED]
- [[robot0_joint_pos_sin_1]] - `defined_in` [EXTRACTED]
- [[robot0_joint_vel_1]] - `defined_in` [EXTRACTED]
- [[robot0_proprio-state_1]] - `defined_in` [EXTRACTED]
- [[test_state_libero.py]] - `shares_data_with` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/LIBERO_Obs_Fixture_Bowls

## 📄 Source

`tests/evasion_tax/metric/fixtures/libero_obs_spatial0.json`

```json
{
  "problem_info": {
    "problem_name": "libero_tabletop_manipulation",
    "domain_name": "robosuite",
    "language_instruction": "pick the akita black bowl between the plate and the ramekin and place it on the plate"
  },
  "language_instruction": "pick the akita black bowl between the plate and the ramekin and place it on the plate",
  "obj_of_interest": [
    "akita_black_bowl_1",
    "plate_1"
  ],
  "obs": {
    "robot0_joint_pos": [
      0.0,
      -0.161037389,
      0.0,
      -2.44459747,
      0.0,
      2.2267522,
      0.7853981633974483
    ],
    "robot0_joint_pos_cos": [
      1.0,
      0.9870614772351539,
      1.0,
      -0.7667744874023851,
      1.0,
      -0.6099170181863868,
      0.7071067811865476
    ],
    "robot0_joint_pos_sin": [
      0.0,
      -0.160342259427625,
      0.0,
      -0.6419165720471856,
      0.0,
      0.7924652868906162,
      0.7071067811865475
    ],
    "robot0_joint_vel": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "robot0_eef_pos": [
      -0.2084646605658239,
      -8.939910281987004e-18,
      1.1732794757296403
    ],
    "robot0_eef_quat": [
      0.9995966048795354,
      0.00024621283188765164,
      -0.02840012048587286,
      -6.99529595901127e-06
    ],
    "robot0_gripper_qpos": [
      0.020833,
      -0.020833
    ],
    "robot0_gripper_qvel": [
      0.0,
      0.0
    ],
    "akita_black_bowl_1_pos": [
      -0.04078457415494016,
      0.19614457121805773,
      0.97
    ],
    "akita_black_bowl_1_quat": [
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475
    ],
    "akita_black_bowl_1_to_robot0_eef_pos": [
      0.17904787289264437,
      -0.19605641019296988,
      0.19343117228241313
    ],
    "akita_black_bowl_1_to_robot0_eef_quat": [
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447
    ],
    "akita_black_bowl_2_pos": [
      -0.1743671686395749,
      0.31169724980163266,
      0.97
    ],
    "akita_black_bowl_2_quat": [
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475
    ],
    "akita_black_bowl_2_to_robot0_eef_pos": [
      0.045737700861786024,
      -0.31167478877999893,
      0.20101563637614406
    ],
    "akita_black_bowl_2_to_robot0_eef_quat": [
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447
    ],
    "cookies_1_pos": [
      0.06484962054062438,
      0.015232368666145558,
      0.97
    ],
    "cookies_1_quat": [
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475
    ],
    "cookies_1_to_robot0_eef_pos": [
      0.2844225333877156,
      -0.015092258595311287,
      0.18743354398393408
    ],
    "cookies_1_to_robot0_eef_quat": [
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447
    ],
    "glazed_rim_porcelain_ramekin_1_pos": [
      -0.18963565772943505,
      0.21298207734192573,
      0.97
    ],
    "glazed_rim_porcelain_ramekin_1_quat": [
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475
    ],
    "glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos": [
      0.030445213684677286,
      -0.21296712705799228,
      0.20188254054257126
    ],
    "glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat": [
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447
    ],
    "plate_1_pos": [
      0.04637735090825516,
      0.20433926159696186,
      0.97
    ],
    "plate_1_quat": [
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475
    ],
    "plate_1_to_robot0_eef_pos": [
      0.26607321970396935,
      -0.20420824125775983,
      0.1884823504237365
    ],
    "plate_1_to_robot0_eef_quat": [
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447
    ],
    "robot0_proprio-state": [
      0.0,
      -0.161037389,
      0.0,
      -2.44459747,
      0.0,
      2.2267522,
      0.7853981633974483,
      1.0,
      0.9870614772351539,
      1.0,
      -0.7667744874023851,
      1.0,
      -0.6099170181863868,
      0.7071067811865476,
      0.0,
      -0.160342259427625,
      0.0,
      -0.6419165720471856,
      0.0,
      0.7924652868906162,
      0.7071067811865475,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.2084646605658239,
      -8.939910281987004e-18,
      1.1732794757296403,
      0.9995966048795354,
      0.00024621283188765164,
      -0.02840012048587286,
      -6.99529595901127e-06,
      0.020833,
      -0.020833,
      0.0,
      0.0
    ],
    "object-state": [
      -0.04078457415494016,
      0.19614457121805773,
      0.97,
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475,
      0.17904787289264437,
      -0.19605641019296988,
      0.19343117228241313,
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447,
      -0.1743671686395749,
      0.31169724980163266,
      0.97,
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475,
      0.045737700861786024,
      -0.31167478877999893,
      0.20101563637614406,
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447,
      0.06484962054062438,
      0.015232368666145558,
      0.97,
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475,
      0.2844225333877156,
      -0.015092258595311287,
      0.18743354398393408,
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447,
      -0.18963565772943505,
      0.21298207734192573,
      0.97,
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475,
      0.030445213684677286,
      -0.21296712705799228,
      0.20188254054257126,
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447,
      0.04637735090825516,
      0.20433926159696186,
      0.97,
      0.0,
      0.0,
      0.7071067811865476,
      0.7071067811865475,
      0.26607321970396935,
      -0.20420824125775983,
      0.1884823504237365,
      0.7069957256317139,
      -0.706647515296936,
      -0.020076973363757133,
      0.0200868658721447
    ]
  }
}
```

