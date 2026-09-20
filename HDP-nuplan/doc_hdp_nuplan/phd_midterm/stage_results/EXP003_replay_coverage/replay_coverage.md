# EXP003｜Replay覆盖重建

时间：09月17日 01:28（北京时间）。

依据当前单卡DistributedSampler默认seed=0、epoch=0和FIFO1024重建；不是从训练日志观测到的抽样序列。
本轮逐文件读取10k NPZ的log_name、scenario_type、token，未修改缓存。

全部日志数：44；保留日志数：44。
保留每log最少/最多：12/31。
全部类型数：39；保留类型数：31。
类别分布总变差距离：0.059555（描述性，不是显著性检验）。
均匀有放回抽1000次的期望唯一组数：638.542；实际未记录。

|类型|完整10k|保留1024|完整比例|保留比例|
|---|---:|---:|---:|---:|
|following_lane_with_slow_lead|24|7|0.2400%|0.6836%|
|following_lane_without_lead|34|4|0.3400%|0.3906%|
|high_lateral_acceleration|3|1|0.0300%|0.0977%|
|high_magnitude_speed|481|56|4.8100%|5.4688%|
|low_magnitude_speed|83|16|0.8300%|1.5625%|
|medium_magnitude_speed|602|61|6.0200%|5.9570%|
|near_barrier_on_driveable|7|1|0.0700%|0.0977%|
|near_construction_zone_sign|12|0|0.1200%|0.0000%|
|near_high_speed_vehicle|80|11|0.8000%|1.0742%|
|near_long_vehicle|66|9|0.6600%|0.8789%|
|near_multiple_vehicles|41|2|0.4100%|0.1953%|
|near_pedestrian_on_crosswalk|22|2|0.2200%|0.1953%|
|near_trafficcone_on_driveable|17|2|0.1700%|0.1953%|
|on_carpark|6|0|0.0600%|0.0000%|
|on_intersection|56|1|0.5600%|0.0977%|
|on_pickup_dropoff|94|15|0.9400%|1.4648%|
|on_stopline_crosswalk|22|2|0.2200%|0.1953%|
|on_stopline_stop_sign|86|13|0.8600%|1.2695%|
|on_stopline_traffic_light|24|2|0.2400%|0.1953%|
|on_traffic_light_intersection|52|5|0.5200%|0.4883%|
|starting_protected_cross_turn|1|1|0.0100%|0.0977%|
|starting_right_turn|3|1|0.0300%|0.0977%|
|starting_unprotected_cross_turn|3|1|0.0300%|0.0977%|
|starting_unprotected_noncross_turn|2|0|0.0200%|0.0000%|
|stationary|2317|207|23.1700%|20.2148%|
|stationary_at_traffic_light_with_lead|46|5|0.4600%|0.4883%|
|stationary_at_traffic_light_without_lead|330|38|3.3000%|3.7109%|
|stationary_in_traffic|726|93|7.2600%|9.0820%|
|stopping_at_crosswalk|2|0|0.0200%|0.0000%|
|stopping_at_stop_sign_no_crosswalk|1|0|0.0100%|0.0000%|
|stopping_at_stop_sign_without_lead|1|0|0.0100%|0.0000%|
|stopping_at_traffic_light_without_lead|3|0|0.0300%|0.0000%|
|stopping_with_lead|1|0|0.0100%|0.0000%|
|traversing_crosswalk|49|3|0.4900%|0.2930%|
|traversing_intersection|331|32|3.3100%|3.1250%|
|traversing_pickup_dropoff|706|67|7.0600%|6.5430%|
|traversing_traffic_light_intersection|1064|103|10.6400%|10.0586%|
|unknown|2596|262|25.9600%|25.5859%|
|waiting_for_pedestrian_to_cross|6|1|0.0600%|0.0977%|

结论边界：保留覆盖不是实际唯一更新覆盖；balanced_logs配额在FIFO后不保证保留。
训练seed42/2026不改变默认sampler seed0，因此在同manifest和epoch下保留集合相同；
候选噪声、回放抽样与扩散加噪仍受训练seed影响。不能把类型分布变化直接归因为闭环退化。
