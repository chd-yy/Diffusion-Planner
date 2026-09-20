# EXP001｜归档 Test14 成对统计与退化分析

生成时间：09月16日 23:32（北京时间）。

这是已有实验的二次分析，不是新训练或新闭环实验。差值均为候选减基线。
按 log 整组有放回重采样，保留组内配对，报告 scenario-weighted 均值差的95%百分位区间。
区间仅描述观测日志间的变异；未覆盖训练/推理随机性、测试集选择偏差或多重比较。
不构成高置信安全保证，不授权替换B10。hard/random分别报告，不合并重复token。

重采样次数：10000；随机种子：3407。

## test14-hard

### 旧RL减B10

272 场景，160 日志。

|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|
|---|---:|---:|---:|---|---|
|score|0.705304|0.702422|-0.002882|[-0.015370, +0.007493]|116/48/108|
|no_ego_at_fault_collisions|0.867647|0.860294|-0.007353|[-0.023077, +0.007018]|1/3/268|
|drivable_area_compliance|0.944853|0.941176|-0.003676|[-0.012097, +0.000000]|0/1/271|
|ego_is_making_progress|0.948529|0.948529|+0.000000|[+0.000000, +0.000000]|0/0/272|
|driving_direction_compliance|0.985294|0.985294|+0.000000|[+0.000000, +0.000000]|0/0/272|
|ego_progress_along_expert_route|0.819141|0.827959|+0.008818|[+0.006560, +0.011309]|149/41/82|
|time_to_collision_within_bound|0.761029|0.761029|+0.000000|[-0.015385, +0.014390]|2/2/268|
|speed_limit_compliance|0.974670|0.972901|-0.001769|[-0.002610, -0.001062]|14/45/213|
|ego_is_comfortable|0.977941|0.974265|-0.003676|[-0.012048, +0.000000]|0/1/271|

最差4场占全部score下降量：97.17%。

|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|
|---|---:|---:|---:|---:|
|changing_lane_to_left|4|+0.000549|+0.001758|+0.000000|
|changing_lane_to_right|8|+0.000094|+0.001415|+0.000000|
|following_lane_with_lead|9|+0.001843|+0.006037|+0.000000|
|high_lateral_acceleration|11|-0.027640|+0.001435|-0.090909|
|high_magnitude_speed|15|+0.020125|-0.007812|+0.000000|
|low_magnitude_speed|5|+0.004518|+0.014459|+0.000000|
|medium_magnitude_speed|6|+0.000978|+0.003738|+0.000000|
|near_high_speed_vehicle|2|-0.000765|-0.002449|+0.000000|
|near_long_vehicle|8|+0.000672|+0.002152|+0.000000|
|near_multiple_vehicles|18|+0.001720|+0.005934|+0.000000|
|on_carpark|1|+0.000000|+0.000000|+0.000000|
|on_intersection|1|+0.002834|+0.009068|+0.000000|
|on_pickup_dropoff|8|-0.084638|+0.024040|-0.125000|
|on_stopline_stop_sign|3|+0.001849|+0.012023|+0.000000|
|on_traffic_light_intersection|3|+0.015887|+0.050837|+0.000000|
|starting_protected_cross_turn|3|+0.004955|+0.013091|+0.000000|
|starting_protected_noncross_turn|1|+0.000000|+0.000000|+0.000000|
|starting_right_turn|5|+0.000240|+0.004042|+0.000000|
|starting_unprotected_cross_turn|4|+0.003844|+0.021075|+0.000000|
|starting_unprotected_noncross_turn|2|+0.001297|+0.004149|+0.000000|
|stationary|15|-0.010167|+0.016452|+0.000000|
|stationary_in_traffic|20|+0.004577|+0.014596|+0.000000|
|stopping_with_lead|20|+0.004478|+0.014331|+0.000000|
|traversing_crosswalk|3|+0.009236|+0.029556|+0.000000|
|traversing_intersection|7|+0.001290|+0.006765|+0.000000|
|traversing_pickup_dropoff|20|+0.015247|+0.006579|+0.000000|
|traversing_traffic_light_intersection|50|-0.015100|+0.002147|+0.000000|
|waiting_for_pedestrian_to_cross|20|+0.005363|+0.023549|+0.000000|

新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。

### 修复RL减B10

272 场景，160 日志。

|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|
|---|---:|---:|---:|---|---|
|score|0.705304|0.704216|-0.001088|[-0.011300, +0.009803]|25/145/102|
|no_ego_at_fault_collisions|0.867647|0.863971|-0.003676|[-0.016598, +0.007968]|1/2/269|
|drivable_area_compliance|0.944853|0.948529|+0.003676|[+0.000000, +0.012097]|1/0/271|
|ego_is_making_progress|0.948529|0.948529|+0.000000|[+0.000000, +0.000000]|0/0/272|
|driving_direction_compliance|0.985294|0.985294|+0.000000|[+0.000000, +0.000000]|0/0/272|
|ego_progress_along_expert_route|0.819141|0.807828|-0.011313|[-0.012837, -0.009905]|5/191/76|
|time_to_collision_within_bound|0.761029|0.772059|+0.011029|[-0.003846, +0.027491]|4/1/267|
|speed_limit_compliance|0.974670|0.976665|+0.001995|[+0.001262, +0.002921]|57/1/214|
|ego_is_comfortable|0.977941|0.970588|-0.007353|[-0.018939, +0.000000]|0/2/270|

最差4场占全部score下降量：66.93%。

|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|
|---|---:|---:|---:|---:|
|changing_lane_to_left|4|+0.077636|-0.001564|+0.000000|
|changing_lane_to_right|8|-0.003278|-0.011913|+0.000000|
|following_lane_with_lead|9|-0.037385|-0.009205|+0.000000|
|high_lateral_acceleration|11|+0.026781|-0.009891|+0.000000|
|high_magnitude_speed|15|+0.019568|-0.008641|+0.000000|
|low_magnitude_speed|5|-0.003057|-0.009783|+0.000000|
|medium_magnitude_speed|6|-0.001069|-0.003574|+0.000000|
|near_high_speed_vehicle|2|-0.003564|-0.011404|+0.000000|
|near_long_vehicle|8|-0.001625|-0.013799|+0.000000|
|near_multiple_vehicles|18|-0.002247|-0.008223|+0.000000|
|on_carpark|1|+0.000000|+0.000000|+0.000000|
|on_intersection|1|-0.009636|-0.030837|+0.000000|
|on_pickup_dropoff|8|+0.039566|-0.011691|+0.000000|
|on_stopline_stop_sign|3|-0.000923|-0.009530|+0.000000|
|on_traffic_light_intersection|3|-0.006173|-0.019754|+0.000000|
|starting_protected_cross_turn|3|-0.002093|-0.008695|+0.000000|
|starting_protected_noncross_turn|1|+0.000000|+0.000000|+0.000000|
|starting_right_turn|5|-0.001191|-0.010816|+0.000000|
|starting_unprotected_cross_turn|4|-0.002278|-0.014157|+0.000000|
|starting_unprotected_noncross_turn|2|-0.003900|-0.012481|+0.000000|
|stationary|15|-0.002090|-0.006974|+0.000000|
|stationary_in_traffic|20|-0.005611|-0.017979|+0.000000|
|stopping_with_lead|20|-0.003565|-0.011408|+0.000000|
|traversing_crosswalk|3|-0.008093|-0.030899|+0.000000|
|traversing_intersection|7|-0.107383|-0.009015|-0.142857|
|traversing_pickup_dropoff|20|+0.046759|-0.004359|+0.050000|
|traversing_traffic_light_intersection|50|-0.017893|-0.014458|-0.020000|
|waiting_for_pedestrian_to_cross|20|-0.002789|-0.013526|+0.000000|

新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。

### 修复RL减旧RL

272 场景，160 日志。

|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|
|---|---:|---:|---:|---|---|
|score|0.702422|0.704216|+0.001794|[-0.013501, +0.019419]|25/145/102|
|no_ego_at_fault_collisions|0.860294|0.863971|+0.003676|[-0.015209, +0.023529]|4/3/265|
|drivable_area_compliance|0.941176|0.948529|+0.007353|[+0.000000, +0.024194]|2/0/270|
|ego_is_making_progress|0.948529|0.948529|+0.000000|[+0.000000, +0.000000]|0/0/272|
|driving_direction_compliance|0.985294|0.985294|+0.000000|[+0.000000, +0.000000]|0/0/272|
|ego_progress_along_expert_route|0.827959|0.807828|-0.020131|[-0.023562, -0.017201]|6/191/75|
|time_to_collision_within_bound|0.761029|0.772059|+0.011029|[-0.007547, +0.031128]|5/2/265|
|speed_limit_compliance|0.972901|0.976665|+0.003764|[+0.002374, +0.005478]|56/2/214|
|ego_is_comfortable|0.974265|0.970588|-0.003676|[-0.016502, +0.007843]|1/2/269|

最差4场占全部score下降量：54.35%。

|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|
|---|---:|---:|---:|---:|
|changing_lane_to_left|4|+0.077087|-0.003322|+0.000000|
|changing_lane_to_right|8|-0.003372|-0.013329|+0.000000|
|following_lane_with_lead|9|-0.039228|-0.015241|+0.000000|
|high_lateral_acceleration|11|+0.054421|-0.011326|+0.090909|
|high_magnitude_speed|15|-0.000557|-0.000828|+0.000000|
|low_magnitude_speed|5|-0.007576|-0.024242|+0.000000|
|medium_magnitude_speed|6|-0.002047|-0.007312|+0.000000|
|near_high_speed_vehicle|2|-0.002798|-0.008955|+0.000000|
|near_long_vehicle|8|-0.002298|-0.015951|+0.000000|
|near_multiple_vehicles|18|-0.003966|-0.014157|+0.000000|
|on_carpark|1|+0.000000|+0.000000|+0.000000|
|on_intersection|1|-0.012470|-0.039905|+0.000000|
|on_pickup_dropoff|8|+0.124204|-0.035732|+0.125000|
|on_stopline_stop_sign|3|-0.002772|-0.021553|+0.000000|
|on_traffic_light_intersection|3|-0.022060|-0.070592|+0.000000|
|starting_protected_cross_turn|3|-0.007048|-0.021787|+0.000000|
|starting_protected_noncross_turn|1|+0.000000|+0.000000|+0.000000|
|starting_right_turn|5|-0.001430|-0.014858|+0.000000|
|starting_unprotected_cross_turn|4|-0.006122|-0.035232|+0.000000|
|starting_unprotected_noncross_turn|2|-0.005197|-0.016631|+0.000000|
|stationary|15|+0.008077|-0.023425|+0.000000|
|stationary_in_traffic|20|-0.010188|-0.032576|+0.000000|
|stopping_with_lead|20|-0.008043|-0.025739|+0.000000|
|traversing_crosswalk|3|-0.017329|-0.060456|+0.000000|
|traversing_intersection|7|-0.108673|-0.015780|-0.142857|
|traversing_pickup_dropoff|20|+0.031512|-0.010938|+0.050000|
|traversing_traffic_light_intersection|50|-0.002793|-0.016605|-0.020000|
|waiting_for_pedestrian_to_cross|20|-0.008152|-0.037076|+0.000000|

新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。

## test14-random

### 旧RL减B10

261 场景，148 日志。

|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|
|---|---:|---:|---:|---|---|
|score|0.824270|0.831146|+0.006876|[-0.006173, +0.020408]|81/57/123|
|no_ego_at_fault_collisions|0.908046|0.911877|+0.003831|[-0.008369, +0.017168]|2/1/258|
|drivable_area_compliance|0.965517|0.957854|-0.007663|[-0.020002, +0.000000]|0/2/259|
|ego_is_making_progress|0.980843|0.988506|+0.007663|[+0.000000, +0.018988]|2/0/259|
|driving_direction_compliance|0.998084|0.998084|+0.000000|[+0.000000, +0.000000]|0/0/261|
|ego_progress_along_expert_route|0.910270|0.915213|+0.004944|[+0.003486, +0.006650]|95/36/130|
|time_to_collision_within_bound|0.869732|0.877395|+0.007663|[-0.007491, +0.023256]|3/1/257|
|speed_limit_compliance|0.966454|0.964413|-0.002042|[-0.002881, -0.001340]|6/53/202|
|ego_is_comfortable|0.977011|0.977011|+0.000000|[+0.000000, +0.000000]|0/0/261|

最差4场占全部score下降量：94.08%。

|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|
|---|---:|---:|---:|---:|
|changing_lane_to_left|2|+0.000402|+0.004106|+0.000000|
|changing_lane_to_right|8|-0.000318|-0.001016|+0.000000|
|following_lane_with_lead|7|+0.001170|+0.004202|+0.000000|
|high_lateral_acceleration|11|+0.002130|+0.008246|+0.000000|
|high_magnitude_speed|17|+0.018096|-0.001428|+0.000000|
|low_magnitude_speed|12|+0.002475|+0.009672|+0.000000|
|medium_magnitude_speed|3|+0.000000|+0.002045|+0.000000|
|near_high_speed_vehicle|2|-0.000980|-0.003135|+0.000000|
|near_long_vehicle|6|+0.004292|+0.013115|+0.000000|
|near_multiple_vehicles|17|+0.000472|+0.001621|+0.000000|
|on_carpark|1|+0.000000|+0.000000|+0.000000|
|on_pickup_dropoff|5|-0.001041|+0.000117|+0.000000|
|on_stopline_stop_sign|1|+0.000000|+0.015886|+0.000000|
|on_traffic_light_intersection|4|+0.008579|+0.028701|+0.000000|
|starting_right_turn|4|+0.002143|+0.006858|+0.000000|
|starting_unprotected_cross_turn|2|+0.001561|+0.013271|+0.000000|
|starting_unprotected_noncross_turn|4|+0.001202|+0.003847|+0.000000|
|stationary|14|-0.014136|+0.006249|+0.000000|
|stationary_in_traffic|20|+0.001115|+0.003517|+0.000000|
|stopping_with_lead|20|+0.000922|+0.002950|+0.000000|
|traversing_crosswalk|4|+0.001145|+0.003712|+0.000000|
|traversing_intersection|8|+0.135075|+0.009607|+0.000000|
|traversing_pickup_dropoff|20|-0.028761|+0.006941|+0.000000|
|traversing_traffic_light_intersection|49|+0.000145|+0.002074|+0.000000|
|waiting_for_pedestrian_to_cross|20|+0.049501|+0.011129|+0.050000|

新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。

### 修复RL减B10

261 场景，148 日志。

|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|
|---|---:|---:|---:|---|---|
|score|0.824270|0.830803|+0.006533|[-0.000593, +0.015468]|32/112/117|
|no_ego_at_fault_collisions|0.908046|0.919540|+0.011494|[+0.000000, +0.025926]|3/0/258|
|drivable_area_compliance|0.965517|0.969349|+0.003831|[+0.000000, +0.012712]|1/0/260|
|ego_is_making_progress|0.980843|0.980843|+0.000000|[+0.000000, +0.000000]|0/0/261|
|driving_direction_compliance|0.998084|0.998084|+0.000000|[+0.000000, +0.000000]|0/0/261|
|ego_progress_along_expert_route|0.910270|0.902460|-0.007810|[-0.009313, -0.006523]|1/140/120|
|time_to_collision_within_bound|0.869732|0.873563|+0.003831|[+0.000000, +0.011811]|1/0/260|
|speed_limit_compliance|0.966454|0.968924|+0.002470|[+0.001654, +0.003444]|57/1/203|
|ego_is_comfortable|0.977011|0.977011|+0.000000|[+0.000000, +0.000000]|0/0/261|

最差4场占全部score下降量：10.75%。

|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|
|---|---:|---:|---:|---:|
|changing_lane_to_left|2|+0.000857|-0.001674|+0.000000|
|changing_lane_to_right|8|-0.004511|-0.014435|+0.000000|
|following_lane_with_lead|7|-0.001534|-0.005837|+0.000000|
|high_lateral_acceleration|11|-0.001589|-0.007197|+0.000000|
|high_magnitude_speed|17|+0.016664|-0.005569|+0.000000|
|low_magnitude_speed|12|-0.001550|-0.007839|+0.000000|
|medium_magnitude_speed|3|+0.196775|-0.002060|+0.333333|
|near_high_speed_vehicle|2|-0.003635|-0.011631|+0.000000|
|near_long_vehicle|6|+0.096993|-0.006447|+0.166667|
|near_multiple_vehicles|17|-0.001473|-0.006372|+0.000000|
|on_carpark|1|+0.000000|+0.000000|+0.000000|
|on_pickup_dropoff|5|-0.000383|-0.002485|+0.000000|
|on_stopline_stop_sign|1|+0.000000|-0.015763|+0.000000|
|on_traffic_light_intersection|4|-0.004888|-0.016434|+0.000000|
|starting_right_turn|4|-0.001861|-0.005957|+0.000000|
|starting_unprotected_cross_turn|2|-0.001176|-0.012189|+0.000000|
|starting_unprotected_noncross_turn|4|-0.001981|-0.006338|+0.000000|
|stationary|14|+0.047101|-0.002243|+0.071429|
|stationary_in_traffic|20|-0.001071|-0.003448|+0.000000|
|stopping_with_lead|20|-0.000344|-0.001100|+0.000000|
|traversing_crosswalk|4|-0.002758|-0.008849|+0.000000|
|traversing_intersection|8|-0.001475|-0.015718|+0.000000|
|traversing_pickup_dropoff|20|+0.000088|-0.004943|+0.000000|
|traversing_traffic_light_intersection|49|-0.003270|-0.012563|+0.000000|
|waiting_for_pedestrian_to_cross|20|-0.002387|-0.013449|+0.000000|

新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。

### 修复RL减旧RL

261 场景，148 日志。

|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|
|---|---:|---:|---:|---|---|
|score|0.831146|0.830803|-0.000343|[-0.015837, +0.015310]|30/118/113|
|no_ego_at_fault_collisions|0.911877|0.919540|+0.007663|[-0.010676, +0.027132]|4/2/255|
|drivable_area_compliance|0.957854|0.969349|+0.011494|[+0.000000, +0.026432]|3/0/258|
|ego_is_making_progress|0.988506|0.980843|-0.007663|[-0.018988, +0.000000]|0/2/259|
|driving_direction_compliance|0.998084|0.998084|+0.000000|[+0.000000, +0.000000]|0/0/261|
|ego_progress_along_expert_route|0.915213|0.902460|-0.012754|[-0.015496, -0.010363]|1/140/120|
|time_to_collision_within_bound|0.877395|0.873563|-0.003831|[-0.017621, +0.008197]|1/2/258|
|speed_limit_compliance|0.964413|0.968924|+0.004512|[+0.003011, +0.006307]|58/1/202|
|ego_is_comfortable|0.977011|0.977011|+0.000000|[+0.000000, +0.000000]|0/0/261|

最差4场占全部score下降量：77.61%。

|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|
|---|---:|---:|---:|---:|
|changing_lane_to_left|2|+0.000455|-0.005780|+0.000000|
|changing_lane_to_right|8|-0.004194|-0.013419|+0.000000|
|following_lane_with_lead|7|-0.002704|-0.010040|+0.000000|
|high_lateral_acceleration|11|-0.003719|-0.015443|+0.000000|
|high_magnitude_speed|17|-0.001432|-0.004141|+0.000000|
|low_magnitude_speed|12|-0.004025|-0.017511|+0.000000|
|medium_magnitude_speed|3|+0.196775|-0.004104|+0.333333|
|near_high_speed_vehicle|2|-0.002655|-0.008495|+0.000000|
|near_long_vehicle|6|+0.092702|-0.019563|+0.166667|
|near_multiple_vehicles|17|-0.001945|-0.007993|+0.000000|
|on_carpark|1|+0.000000|+0.000000|+0.000000|
|on_pickup_dropoff|5|+0.000658|-0.002601|+0.000000|
|on_stopline_stop_sign|1|+0.000000|-0.031649|+0.000000|
|on_traffic_light_intersection|4|-0.013467|-0.045135|+0.000000|
|starting_right_turn|4|-0.004005|-0.012815|+0.000000|
|starting_unprotected_cross_turn|2|-0.002737|-0.025460|+0.000000|
|starting_unprotected_noncross_turn|4|-0.003183|-0.010185|+0.000000|
|stationary|14|+0.061237|-0.008492|+0.071429|
|stationary_in_traffic|20|-0.002185|-0.006965|+0.000000|
|stopping_with_lead|20|-0.001266|-0.004050|+0.000000|
|traversing_crosswalk|4|-0.003903|-0.012560|+0.000000|
|traversing_intersection|8|-0.136551|-0.025326|+0.000000|
|traversing_pickup_dropoff|20|+0.028849|-0.011884|+0.000000|
|traversing_traffic_light_intersection|49|-0.003414|-0.014636|+0.000000|
|waiting_for_pedestrian_to_cross|20|-0.051888|-0.024579|-0.050000|

新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。

## 来源与重现

输入摘要见同名JSON（含SHA256、脚本hash和版本）。

- [test14-hard_three_models.json](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/test14-hard_three_models.json)：199262dd2e50dcc8081bd5a4fe4217d076ac454927e10ffc3cb75ff9c7ca8f56
- [test14-random_three_models.json](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/test14-random_three_models.json)：3e4e545db4576de743036931e893117a31bb7f7c4fe5e8e368cd5dacda571f8c
- [full_comparison.json](/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/rl_safety_geometry_test14_seed0/full_comparison.json)：6eaec01bb7b7333678dd975fc3e7c91c0f4364f2aac7e77997fdfdb9d01d3967
