#import matplotlib.pyplot as plt
#
### Data for Original Model (LR=0.2)
##original_train_loss = [2.0496, 2.0352, 1.8943, 1.8820, 1.8235, 1.6981, 1.6331, 1.6030, 1.5612, 1.5111, 1.4802, 1.4532, 1.4091, 1.4164, 1.3914, 1.3633, 1.3258, 1.3520, 1.2808, 1.2748, 1.3117, 1.2912, 1.2397, 1.1991, 1.1836, 1.1906, 1.2014, 1.1555, 1.3176, 1.1673]
##original_test_loss = [2.0291, 2.0536, 1.8900, 1.8497, 1.8162, 1.6608, 1.6054, 1.5488, 1.5320, 1.5106, 1.4540, 1.4357, 1.4001, 1.4097, 1.3937, 1.3631, 1.3500, 1.3554, 1.2874, 1.3006, 1.3270, 1.3196, 1.2601, 1.2482, 1.2166, 1.2094, 1.2448, 1.2083, 1.3094, 1.1901]
##orig_test_acc = [0.2472, 0.2384, 0.3034, 0.3196, 0.3322, 0.4095, 0.4249, 0.4409, 0.4503, 0.4579, 0.4779, 0.4791, 0.4928, 0.4966, 0.5043, 0.5075, 0.5140, 0.5135, 0.5434, 0.5293, 0.5138, 0.5258, 0.5425, 0.5540, 0.5659, 0.5670, 0.5538, 0.5606, 0.5379, 0.5744]
##
### Data for New Model (Without Wup/Wv, LR=0.2)
##new_train_loss = [2.0966, 2.0358, 1.9782, 1.9423, 1.9698, 1.8654, 1.8451, 1.7562, 1.7193, 1.7613, 1.6838, 1.6274, 1.6367, 1.5566, 1.5772, 1.5243, 1.5023, 1.4847, 1.4952, 1.4769, 1.4560, 1.4596, 1.4328, 1.4363, 1.4111, 1.3842, 1.3695, 1.3595, 1.3824, 1.3511]
##new_test_loss = [2.0983, 2.0164, 1.9880, 1.9237, 1.9236, 1.8393, 1.7970, 1.7173, 1.6830, 1.7514, 1.6201, 1.6272, 1.6037, 1.5173, 1.5322, 1.5137, 1.4760, 1.4706, 1.4710, 1.4533, 1.4256, 1.4193, 1.4249, 1.4405, 1.4150, 1.3655, 1.3614, 1.3480, 1.3634, 1.3405]
##new_test_acc = [0.2237, 0.2596, 0.2644, 0.2923, 0.2904, 0.3215, 0.3406, 0.3689, 0.3894, 0.3835, 0.4176, 0.4130, 0.4268, 0.4615, 0.4541, 0.4584, 0.4683, 0.4691, 0.4752, 0.4767, 0.4888, 0.4897, 0.4929, 0.4869, 0.4913, 0.5101, 0.5092, 0.5126, 0.5145, 0.5161]
#
#
#
### --- DATA FOR ORIGINAL MODEL (Standard) ---
##original_train_loss = [
##    1.9988, 1.9078, 1.7814, 1.6915, 1.6441, 1.5492, 1.5112, 1.4992, 1.4668, 1.5202,
##    1.3650, 1.3281, 1.3243, 1.2898, 1.2749, 1.3283, 1.2449, 1.2344, 1.2162, 1.2843,
##    1.1947, 1.2354, 1.1316, 1.1081, 1.1553, 1.1152, 1.1134, 1.0463, 1.1470, 1.0599
##]
##original_test_loss = [
##    1.9782, 1.9225, 1.7360, 1.6597, 1.6347, 1.5128, 1.4921, 1.4533, 1.4454, 1.5037,
##    1.3459, 1.3239, 1.3201, 1.2903, 1.2967, 1.3364, 1.2652, 1.2463, 1.2565, 1.3020,
##    1.2510, 1.2949, 1.1749, 1.1726, 1.1934, 1.1452, 1.1608, 1.1161, 1.1999, 1.1162
##]
##orig_test_acc = [
##    0.2632, 0.2911, 0.3663, 0.3995, 0.4113, 0.4608, 0.4660, 0.4713, 0.4783, 0.4628,
##    0.5157, 0.5262, 0.5212, 0.5381, 0.5353, 0.5186, 0.5447, 0.5486, 0.5456, 0.5345,
##    0.5467, 0.5429, 0.5800, 0.5784, 0.5786, 0.5933, 0.5786, 0.6053, 0.5758, 0.6042
##]
##
### --- DATA FOR SIMPLIFIED MODEL (Without Wup and Wv) ---
##new_train_loss = [
##    2.0574, 1.9548, 1.9264, 1.8484, 1.7754, 1.7136, 1.6095, 1.6799, 1.5725, 1.5656,
##    1.5047, 1.5148, 1.4556, 1.4830, 1.4510, 1.4287, 1.3871, 1.3783, 1.3796, 1.3758,
##    1.3619, 1.3536, 1.3503, 1.3142, 1.2998, 1.2745, 1.2572, 1.2470, 1.3089, 1.2432
##]
##new_test_loss = [
##    2.0453, 1.9189, 1.9622, 1.7849, 1.7651, 1.6367, 1.5847, 1.5968, 1.5318, 1.5356,
##    1.4556, 1.4997, 1.4347, 1.4495, 1.4213, 1.3996, 1.3771, 1.3723, 1.3760, 1.3651,
##    1.3568, 1.3569, 1.3542, 1.3362, 1.3249, 1.2826, 1.2762, 1.2558, 1.3081, 1.2660
##]
##new_test_acc = [
##    0.2416, 0.2954, 0.2748, 0.3449, 0.3669, 0.4089, 0.4268, 0.4140, 0.4440, 0.4426,
##    0.4749, 0.4599, 0.4828, 0.4818, 0.4957, 0.4975, 0.5032, 0.5084, 0.5161, 0.5047,
##    0.5187, 0.5159, 0.5194, 0.5265, 0.5246, 0.5377, 0.5426, 0.5469, 0.5336, 0.5477
##]
#
## --- 1. DATA FOR ORIGINAL MODEL (Standard) ---
#original_train_loss = [
#    1.9844, 1.8650, 1.7509, 1.6356, 1.5947, 1.5247, 1.4812, 1.4616, 1.3858, 1.3350,
#    1.3271, 1.3079, 1.2962, 1.2350, 1.3171, 1.3234, 1.2000, 1.2661, 1.2103, 1.2457,
#    1.1429, 1.4311, 1.0553, 1.0428, 1.1216, 1.0473, 1.0541, 0.9724, 1.0223, 0.9884
#]
#original_test_loss = [
#    1.9687, 1.8475, 1.6937, 1.6073, 1.5759, 1.4907, 1.4692, 1.4359, 1.3787, 1.3513,
#    1.3163, 1.3103, 1.2914, 1.2431, 1.3331, 1.3391, 1.2252, 1.2948, 1.2513, 1.2941,
#    1.1869, 1.5290, 1.1107, 1.1129, 1.2095, 1.0872, 1.1291, 1.0489, 1.0622, 1.0473
#]
#orig_test_acc = [
#    0.2658, 0.3219, 0.3931, 0.4244, 0.4371, 0.4667, 0.4711, 0.4828, 0.5048, 0.5074,
#    0.5326, 0.5224, 0.5341, 0.5585, 0.5299, 0.5299, 0.5592, 0.5341, 0.5517, 0.5317,
#    0.5675, 0.4886, 0.6003, 0.6051, 0.5756, 0.6126, 0.5935, 0.6238, 0.6264, 0.6307
#]
#
## --- 2. DATA FOR WITHOUT Wup AND Wv MODEL ---
#new_train_loss = [
#    2.0599, 1.9671, 1.8625, 1.7708, 1.6744, 1.8107, 1.5577, 1.5877, 1.5132, 1.4758,
#    1.4792, 1.4809, 1.3994, 1.4316, 1.4253, 1.3786, 1.3310, 1.3528, 1.3197, 1.3411,
#    1.3395, 1.3098, 1.3033, 1.2724, 1.2698, 1.2284, 1.2064, 1.2160, 1.2522, 1.1967
#]
#new_test_loss = [
#    2.0186, 1.9063, 1.8590, 1.7108, 1.6510, 1.8094, 1.5127, 1.5259, 1.4726, 1.4680,
#    1.4262, 1.4767, 1.3896, 1.4225, 1.3985, 1.3650, 1.3390, 1.3666, 1.3444, 1.3357,
#    1.3483, 1.3210, 1.3167, 1.3076, 1.3180, 1.2502, 1.2392, 1.2315, 1.2640, 1.2447
#]
#new_test_acc = [
#    0.2543, 0.2955, 0.3206, 0.3806, 0.4001, 0.3717, 0.4514, 0.4412, 0.4680, 0.4759,
#    0.4897, 0.4751, 0.4991, 0.4932, 0.5008, 0.5066, 0.5154, 0.5108, 0.5242, 0.5254,
#    0.5145, 0.5258, 0.5276, 0.5322, 0.5264, 0.5463, 0.5537, 0.5568, 0.5487, 0.5468
#]
#
#
#epochs = range(1, 31)
#
## Create the figure and axis
##fig, ax = plt.subplots(figsize=(12, 8))
#fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
#
## Original Model (Blue)
#ax.plot(epochs, original_train_loss, color='blue', linestyle='-', linewidth=2, label='Original Train Loss')
#ax.plot(epochs, new_train_loss, color='red', linestyle='-', linewidth=2, label='Without Wup and Wv Train Loss')
#
## New Model (Red)
#ax.plot(epochs, original_test_loss, color='blue', linestyle=':', linewidth=2, label='Original Test Loss')
#ax.plot(epochs, new_test_loss, color='red', linestyle=':', linewidth=2, label='Without Wup and Wv Test Loss')
#
## --- MAKE TEXT BIGGER ---
#ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
#ax.set_ylabel('Loss', fontsize=16, fontweight='bold')
#ax.set_title('Train and test loss', fontsize=18, fontweight='bold')
#ax.tick_params(axis='both', which='major', labelsize=14) # Size of numbers on axes
#
## --- MAKE BORDER (SPINES) THICKER ---
#for spine in ax.spines.values():
#    spine.set_linewidth(2.5) # Thicker outer border
#
## --- ADD LEGEND AND GRID ---
#ax.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='black')
#ax.grid(True, which='both', linestyle='--', alpha=0.5)
#
## --- RIGHT PLOT: Test Accuracy Comparison ---
#ax2.plot(epochs, orig_test_acc, color='blue', linestyle='-', linewidth=3, label='Original Test Acc')
#ax2.plot(epochs, new_test_acc, color='red', linestyle='-', linewidth=3, label='Without Wup and Wv Test Acc')
#
#ax2.set_xlabel('Epoch', fontsize=16, fontweight='bold')
#ax2.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
#ax2.set_title('Test accuracy', fontsize=18, fontweight='bold')
#ax2.tick_params(axis='both', which='major', labelsize=14)
#for spine in ax2.spines.values():
#    spine.set_linewidth(2.5) # Thicker border
#ax2.legend(fontsize=12, loc='lower right', frameon=True, edgecolor='black')
#ax2.grid(True, which='both', linestyle='--', alpha=0.5)
#
#plt.tight_layout()
#plt.show()




import matplotlib.pyplot as plt

## --- DATA FOR DEPTH 3 (Orange) ---
#lrs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#
#train_loss_3 = [1.6670, 1.4562, 1.4215, 1.4178, 1.4172, 1.4028, 1.3988, 1.3822, 1.3515, 1.3468]
#test_loss_3  = [1.6097, 1.4451, 1.3998, 1.3909, 1.3802, 1.3656, 1.3579, 1.3445, 1.3232, 1.3249]
#train_acc_3  = [0.3797, 0.4630, 0.4782, 0.4831, 0.4864, 0.4916, 0.4935, 0.4994, 0.5068, 0.5099]
#test_acc_3   = [0.4055, 0.4782, 0.4915, 0.4996, 0.5055, 0.5087, 0.5153, 0.5208, 0.5256, 0.5222]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.5535, 1.4802, 1.4446, 1.3985, 1.3601, 1.3221, 1.2839, 1.2815, 1.2821, 1.2915]
#test_loss_6  = [1.5393, 1.3957, 1.3914, 1.3404, 1.3529, 1.3137, 1.2789, 1.2800, 1.2767, 1.2976]
#train_acc_6  = [0.4272, 0.4509, 0.4652, 0.4866, 0.5054, 0.5208, 0.5323, 0.5390, 0.5400, 0.5369]
#test_acc_6   = [0.4368, 0.4861, 0.4941, 0.5118, 0.5153, 0.5285, 0.5365, 0.5374, 0.5363, 0.5360]
#
## --- DATA FOR DEPTH 9 (Purple) ---
## Extracted from your text, starting from LR 0.2 to match X-axis
#train_loss_9 = [1.5153, 1.4166, 1.3759, 1.3361, 1.3168, 1.3297, 1.2935, 1.2874, 1.2661, 1.2822]
#test_loss_9  = [1.4846, 1.3958, 1.3517, 1.3138, 1.3048, 1.2929, 1.2895, 1.2852, 1.2747, 1.2808]
#train_acc_9  = [0.4357, 0.4746, 0.4894, 0.5071, 0.5126, 0.5123, 0.5281, 0.5319, 0.5341, 0.5315]
#test_acc_9   = [0.4554, 0.4891, 0.5049, 0.5152, 0.5176, 0.5261, 0.5303, 0.5315, 0.5354, 0.5353]

## --- SHARED X-AXIS ---
#lrs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.3951, 1.3712, 1.2514, 1.2507, 1.2402, 1.2146, 1.2368, 1.2175, 1.2431, 1.2306]
#test_loss_3  = [1.3905, 1.3691, 1.2551, 1.2603, 1.2450, 1.2164, 1.2336, 1.2238, 1.2430, 1.2377]
#train_acc_3  = [0.4884, 0.4981, 0.5487, 0.5457, 0.5482, 0.5592, 0.5525, 0.5582, 0.5491, 0.5536]
#test_acc_3   = [0.4902, 0.5046, 0.5451, 0.5384, 0.5464, 0.5636, 0.5545, 0.5553, 0.5457, 0.5489]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.3756, 1.2961, 1.2495, 1.2118, 1.1738, 1.1564, 1.1550, 1.1683, 1.1512, 1.1273]
#test_loss_6  = [1.3677, 1.2940, 1.2579, 1.2101, 1.1851, 1.1682, 1.1672, 1.1802, 1.1660, 1.1670]
#train_acc_6  = [0.4962, 0.5310, 0.5461, 0.5597, 0.5744, 0.5804, 0.5802, 0.5767, 0.5817, 0.5925]
#test_acc_6   = [0.5055, 0.5315, 0.5471, 0.5660, 0.5700, 0.5766, 0.5768, 0.5725, 0.5785, 0.5803]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.3789, 1.3219, 1.2802, 1.2596, 1.2555, 1.2096, 1.2033, 1.2037, 1.1407, 1.1720]
#test_loss_9  = [1.3391, 1.3163, 1.2725, 1.2512, 1.2478, 1.2144, 1.2265, 1.2277, 1.1804, 1.2029]
#train_acc_9  = [0.4920, 0.5187, 0.5356, 0.5404, 0.5510, 0.5659, 0.5637, 0.5664, 0.5915, 0.5785]
#test_acc_9   = [0.5130, 0.5252, 0.5404, 0.5496, 0.5525, 0.5596, 0.5603, 0.5594, 0.5774, 0.5660]
#
#
#def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
#    plt.figure(figsize=(10, 6))
#    
#    # Plot Depth 3 (Orange)
#    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=1.5, label='Depth=3')
#    
#    # Plot Depth 6 (Red)
#    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=1.5, label='Depth=6')
#    
#    # Plot Depth 9 (Purple)
#    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=1.5, label='Depth=9')
#    
#    # Axis Labels
#    plt.xlabel('Learning Rate', fontsize=20, fontweight='bold')
#    plt.ylabel(ylabel_text, fontsize=20, fontweight='bold')
#    
#    # Ticks
#    plt.xticks(lrs, fontsize=14)
#    plt.yticks(fontsize=14)
#    
#    # Grid
#    plt.grid(True, linestyle='--', alpha=0.5)
#    
#    # Legend
#    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
#    
#    # Remove dots from legend box for all lines
#    for line in leg.get_lines():
#        line.set_marker("") 
#    
#    # Set Y-axis limit
#    if ylim_range:
#        plt.ylim(ylim_range)
#    
#    # Thicken borders
#    ax = plt.gca()
#    for spine in ax.spines.values():
#        spine.set_linewidth(2.0)
#    
#    plt.tight_layout()
#    plt.show()
#
## --- RENDER GRAPHS ---
#
## 1. Train Loss (Legend at TOP)
#plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.2, 1.8), legend_loc='upper right')
#
## 2. Test Loss (Legend at TOP)
#plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.2, 1.8), legend_loc='upper right')
#
## 3. Train Accuracy (Legend at BOTTOM)
#plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.3, 0.55), legend_loc='lower right')
#
## 4. Test Accuracy (Legend at BOTTOM)
#plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.3, 0.55), legend_loc='lower right')



import matplotlib.pyplot as plt

## --- SHARED X-AXIS ---
#lrs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.3951, 1.3420, 1.2514, 1.2507, 1.2500, 1.2146, 1.2368, 1.2175, 1.2431, 1.2306]
#test_loss_3  = [1.3905, 1.3423, 1.2551, 1.2603, 1.2468, 1.2164, 1.2336, 1.2238, 1.2430, 1.2377]
#valid_loss_3 = [1.3766, 1.3161, 1.2153, 1.2272, 1.2078, 1.1744, 1.1895, 1.1840, 1.2036, 1.1979]
#
#train_acc_3  = [0.4884, 0.5096, 0.5487, 0.5457, 0.5491, 0.5592, 0.5525, 0.5582, 0.5491, 0.5536]
#test_acc_3   = [0.4902, 0.5031, 0.5451, 0.5384, 0.5405, 0.5636, 0.5545, 0.5553, 0.5457, 0.5489]
#valid_acc_3  = [0.4922, 0.5113, 0.5638, 0.5596, 0.5641, 0.5807, 0.5695, 0.5779, 0.5656, 0.5684]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.3756, 1.2961, 1.2495, 1.2118, 1.1738, 1.1564, 1.1550, 1.1683, 1.1512, 1.1538]
#test_loss_6  = [1.3677, 1.2940, 1.2579, 1.2101, 1.1851, 1.1682, 1.1672, 1.1802, 1.1660, 1.1783]
#valid_loss_6 = [1.3479, 1.2630, 1.2318, 1.1794, 1.1449, 1.1323, 1.1354, 1.1454, 1.1388, 1.1448]
#
#train_acc_6  = [0.4962, 0.5310, 0.5461, 0.5597, 0.5744, 0.5804, 0.5802, 0.5767, 0.5817, 0.5841]
#test_acc_6   = [0.5055, 0.5315, 0.5471, 0.5660, 0.5700, 0.5766, 0.5768, 0.5725, 0.5785, 0.5744]
#valid_acc_6  = [0.5133, 0.5440, 0.5554, 0.5800, 0.5889, 0.5955, 0.5948, 0.5903, 0.5937, 0.5808]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.3789, 1.3177, 1.2802, 1.2588, 1.2555, 1.2096, 1.2095, 1.2037, 1.1407, 1.1463]
#test_loss_9  = [1.3391, 1.3087, 1.2725, 1.2527, 1.2478, 1.2144, 1.2182, 1.2277, 1.1804, 1.1971]
#valid_loss_9 = [1.3245, 1.2874, 1.2467, 1.2247, 1.2221, 1.1844, 1.1878, 1.1936, 1.1466, 1.1539]
#
#train_acc_9  = [0.4920, 0.5204, 0.5356, 0.5466, 0.5510, 0.5659, 0.5686, 0.5664, 0.5915, 0.5865]
#test_acc_9   = [0.5130, 0.5246, 0.5404, 0.5447, 0.5525, 0.5596, 0.5595, 0.5594, 0.5774, 0.5636]
#valid_acc_9  = [0.5166, 0.5281, 0.5457, 0.5628, 0.5595, 0.5770, 0.5728, 0.5713, 0.5990, 0.5869]

# --- SHARED X-AXIS ---
lrs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# --- DATA FOR DEPTH 3 (Orange) ---
train_loss_3 = [1.6670, 1.4562, 1.4215, 1.4178, 1.4172, 1.4028, 1.3988, 1.3822, 1.3515, 1.3468]
test_loss_3  = [1.6097, 1.4451, 1.3998, 1.3909, 1.3802, 1.3656, 1.3579, 1.3445, 1.3232, 1.3249]
valid_loss_3 = [1.6134, 1.4323, 1.3853, 1.3803, 1.3713, 1.3570, 1.3488, 1.3332, 1.3088, 1.3084]

train_acc_3  = [0.3797, 0.4630, 0.4782, 0.4831, 0.4864, 0.4916, 0.4935, 0.4994, 0.5068, 0.5099]
test_acc_3   = [0.4055, 0.4782, 0.4915, 0.4996, 0.5055, 0.5087, 0.5153, 0.5208, 0.5256, 0.5222]
valid_acc_3  = [0.4080, 0.4776, 0.4906, 0.4995, 0.5033, 0.5132, 0.5149, 0.5133, 0.5219, 0.5263]

# --- DATA FOR DEPTH 6 (Red) ---
train_loss_6 = [1.5954, 1.4802, 1.4308, 1.3985, 1.3408, 1.3221, 1.2839, 1.2815, 1.2821, 1.2853]
test_loss_6  = [1.5193, 1.3957, 1.3900, 1.3404, 1.3360, 1.3137, 1.2789, 1.2800, 1.2767, 1.2867]
valid_loss_6 = [1.5120, 1.3844, 1.3724, 1.3212, 1.3118, 1.2849, 1.2494, 1.2527, 1.2437, 1.2535]

train_acc_6  = [0.4014, 0.4509, 0.4743, 0.4866, 0.5096, 0.5208, 0.5323, 0.5390, 0.5400, 0.5281]
test_acc_6   = [0.4342, 0.4861, 0.4906, 0.5118, 0.5139, 0.5285, 0.5365, 0.5374, 0.5363, 0.5311]
valid_acc_6  = [0.4473, 0.4946, 0.5011, 0.5250, 0.5185, 0.5411, 0.5501, 0.5520, 0.5536, 0.5435]

# --- DATA FOR DEPTH 9 (Purple) ---
train_loss_9 = [1.5153, 1.4166, 1.3759, 1.3361, 1.3168, 1.3297, 1.2935, 1.2876, 1.2661, 1.2609]
test_loss_9  = [1.4846, 1.3958, 1.3517, 1.3138, 1.3048, 1.2929, 1.2895, 1.2877, 1.2747, 1.2823]
valid_loss_9 = [1.4746, 1.3784, 1.3314, 1.2899, 1.2785, 1.2666, 1.2674, 1.2633, 1.2455, 1.2529]

train_acc_9  = [0.4357, 0.4746, 0.4894, 0.5071, 0.5126, 0.5123, 0.5281, 0.5248, 0.5341, 0.5371]
test_acc_9   = [0.4554, 0.4891, 0.5049, 0.5152, 0.5176, 0.5261, 0.5303, 0.5273, 0.5354, 0.5313]
valid_acc_9  = [0.4628, 0.4953, 0.5038, 0.5297, 0.5336, 0.5361, 0.5421, 0.5383, 0.5371, 0.5390]


def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
    plt.figure(figsize=(10, 6))
    
    # Plot depths with reduced thickness and dot size
    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=1.0, label='Depth=3')
    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=1.0, label='Depth=6')
    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=1.0, label='Depth=9')
    
    # Labeling
    plt.xlabel('Learning Rate', fontsize=18, fontweight='bold')
    plt.ylabel(ylabel_text, fontsize=18, fontweight='bold')
    plt.xticks(lrs, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Legend formatting
    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
    for line in leg.get_lines():
        line.set_marker("") 
    
    # --- ADJUSTED AXIS RANGE ---
    if ylim_range:
        plt.ylim(ylim_range)
    
    # Border styling (Keeping the outer box border intact)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    plt.tight_layout()
    plt.show()

# --- GENERATE PLOTS WITH NEW RANGES ---

# 1. Train Loss (Range: 1.20 - 1.70)
plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.20, 1.70), legend_loc='upper right')

# 2. Test Loss (Range: 1.20 - 1.70)
plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.20, 1.70), legend_loc='upper right')

# 3. Valid Loss (Range: 1.20 - 1.70)
plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', ylim_range=(1.20, 1.70), legend_loc='upper right')

# 4. Train Accuracy (Range: 0.35 - 0.58)
plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')

# 5. Test Accuracy (Range: 0.35 - 0.58)
plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')

# 6. Valid Accuracy (Range: 0.35 - 0.58)
plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')