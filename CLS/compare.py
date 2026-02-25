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

## --- SHARED X-AXIS ---
#lrs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.6670, 1.4562, 1.4215, 1.4178, 1.4172, 1.4028, 1.3988, 1.3822, 1.3515, 1.3468]
#test_loss_3  = [1.6097, 1.4451, 1.3998, 1.3909, 1.3802, 1.3656, 1.3579, 1.3445, 1.3232, 1.3249]
#valid_loss_3 = [1.6134, 1.4323, 1.3853, 1.3803, 1.3713, 1.3570, 1.3488, 1.3332, 1.3088, 1.3084]
#
#train_acc_3  = [0.3797, 0.4630, 0.4782, 0.4831, 0.4864, 0.4916, 0.4935, 0.4994, 0.5068, 0.5099]
#test_acc_3   = [0.4055, 0.4782, 0.4915, 0.4996, 0.5055, 0.5087, 0.5153, 0.5208, 0.5256, 0.5222]
#valid_acc_3  = [0.4080, 0.4776, 0.4906, 0.4995, 0.5033, 0.5132, 0.5149, 0.5133, 0.5219, 0.5263]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.5954, 1.4802, 1.4308, 1.3985, 1.3408, 1.3221, 1.2839, 1.2815, 1.2821, 1.2853]
#test_loss_6  = [1.5193, 1.3957, 1.3900, 1.3404, 1.3360, 1.3137, 1.2789, 1.2800, 1.2767, 1.2867]
#valid_loss_6 = [1.5120, 1.3844, 1.3724, 1.3212, 1.3118, 1.2849, 1.2494, 1.2527, 1.2437, 1.2535]
#
#train_acc_6  = [0.4014, 0.4509, 0.4743, 0.4866, 0.5096, 0.5208, 0.5323, 0.5390, 0.5400, 0.5281]
#test_acc_6   = [0.4342, 0.4861, 0.4906, 0.5118, 0.5139, 0.5285, 0.5365, 0.5374, 0.5363, 0.5311]
#valid_acc_6  = [0.4473, 0.4946, 0.5011, 0.5250, 0.5185, 0.5411, 0.5501, 0.5520, 0.5536, 0.5435]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.5153, 1.4166, 1.3759, 1.3361, 1.3168, 1.3297, 1.2935, 1.2876, 1.2661, 1.2609]
#test_loss_9  = [1.4846, 1.3958, 1.3517, 1.3138, 1.3048, 1.2929, 1.2895, 1.2877, 1.2747, 1.2823]
#valid_loss_9 = [1.4746, 1.3784, 1.3314, 1.2899, 1.2785, 1.2666, 1.2674, 1.2633, 1.2455, 1.2529]
#
#train_acc_9  = [0.4357, 0.4746, 0.4894, 0.5071, 0.5126, 0.5123, 0.5281, 0.5248, 0.5341, 0.5371]
#test_acc_9   = [0.4554, 0.4891, 0.5049, 0.5152, 0.5176, 0.5261, 0.5303, 0.5273, 0.5354, 0.5313]
#valid_acc_9  = [0.4628, 0.4953, 0.5038, 0.5297, 0.5336, 0.5361, 0.5421, 0.5383, 0.5371, 0.5390]
#
#
#def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
#    plt.figure(figsize=(10, 6))
#    
#    # Plot depths with reduced thickness and dot size
#    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=1.0, label='Depth=3')
#    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=1.0, label='Depth=6')
#    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=1.0, label='Depth=9')
#    
#    # Labeling
#    plt.xlabel('Learning Rate', fontsize=18, fontweight='bold')
#    plt.ylabel(ylabel_text, fontsize=18, fontweight='bold')
#    plt.xticks(lrs, fontsize=12)
#    plt.yticks(fontsize=12)
#    plt.grid(True, linestyle='--', alpha=0.5)
#    
#    # Legend formatting
#    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
#    for line in leg.get_lines():
#        line.set_marker("") 
#    
#    # --- ADJUSTED AXIS RANGE ---
#    if ylim_range:
#        plt.ylim(ylim_range)
#    
#    # Border styling (Keeping the outer box border intact)
#    ax = plt.gca()
#    for spine in ax.spines.values():
#        spine.set_linewidth(2.0)
#    
#    plt.tight_layout()
#    plt.show()
#
## --- GENERATE PLOTS WITH NEW RANGES ---
#
## 1. Train Loss (Range: 1.20 - 1.70)
#plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.20, 1.70), legend_loc='upper right')
#
## 2. Test Loss (Range: 1.20 - 1.70)
#plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.20, 1.70), legend_loc='upper right')
#
## 3. Valid Loss (Range: 1.20 - 1.70)
#plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', ylim_range=(1.20, 1.70), legend_loc='upper right')
#
## 4. Train Accuracy (Range: 0.35 - 0.58)
#plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')
#
## 5. Test Accuracy (Range: 0.35 - 0.58)
#plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')
#
## 6. Valid Accuracy (Range: 0.35 - 0.58)
#plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')








#import matplotlib.pyplot as plt
#import numpy as np
#
## --- SHARED X-AXIS ---
#lrs = np.array([0.100, 0.146, 0.213, 0.311, 0.453, 0.662, 0.965, 1.409, 2.056])
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.5877, 1.5069, 1.4428, 1.3800, 1.3502, 1.3278, 1.2539, 1.2266, 1.2821]
#test_loss_3  = [1.5781, 1.4907, 1.4394, 1.3676, 1.3489, 1.3115, 1.2638, 1.2494, 1.2653]
#valid_loss_3 = [1.5749, 1.4890, 1.4317, 1.3546, 1.3250, 1.2815, 1.2297, 1.2149, 1.2303]
#
#train_acc_3  = [0.4052, 0.4376, 0.4682, 0.4942, 0.5054, 0.5139, 0.5473, 0.5551, 0.5334]
#test_acc_3   = [0.4119, 0.4521, 0.4735, 0.4994, 0.5068, 0.5222, 0.5440, 0.5455, 0.5428]
#valid_acc_3  = [0.4215, 0.4523, 0.4703, 0.4986, 0.5085, 0.5250, 0.5581, 0.5656, 0.5631]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.6077, 1.5504, 1.4664, 1.4295, 1.3871, 1.3194, 1.2815, 1.2162, 1.2093]
#test_loss_6  = [1.5765, 1.5145, 1.4503, 1.4167, 1.3708, 1.3185, 1.2757, 1.2234, 1.2165]
#valid_loss_6 = [1.5746, 1.5121, 1.4459, 1.4006, 1.3481, 1.2890, 1.2511, 1.1873, 1.1779]
#
#train_acc_6  = [0.3974, 0.4163, 0.4533, 0.4756, 0.4946, 0.5209, 0.5325, 0.5570, 0.5618]
#test_acc_6   = [0.4088, 0.4419, 0.4676, 0.4852, 0.5039, 0.5226, 0.5352, 0.5581, 0.5584]
#valid_acc_6  = [0.4180, 0.4439, 0.4696, 0.4916, 0.5139, 0.5317, 0.5478, 0.5719, 0.5700]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.6235, 1.5526, 1.4753, 1.4286, 1.3759, 1.3536, 1.3258, 1.2950, 1.2154]
#test_loss_9  = [1.5955, 1.5094, 1.4475, 1.4097, 1.3550, 1.3475, 1.3088, 1.2761, 1.2175]
#valid_loss_9 = [1.5921, 1.5120, 1.4432, 1.3984, 1.3391, 1.3211, 1.2758, 1.2518, 1.1849]
#
#train_acc_9  = [0.3868, 0.4182, 0.4495, 0.4711, 0.4926, 0.5013, 0.5153, 0.5260, 0.5581]
#test_acc_9   = [0.3947, 0.4407, 0.4650, 0.4842, 0.5096, 0.5067, 0.5225, 0.5421, 0.5649]
#valid_acc_9  = [0.3948, 0.4319, 0.4655, 0.4886, 0.5105, 0.5105, 0.5313, 0.5472, 0.5752]
#
#
#def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
#    plt.figure(figsize=(10, 6))
#    
#    # Plot depths with reduced thickness and dot size
#    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=1.0, label='Depth=3')
#    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=1.0, label='Depth=6')
#    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=1.0, label='Depth=9')
#    
#    # Apply Base-2 Logarithmic Scale to X-Axis
#    plt.xscale('log', base=2)
#    
#    # Labeling
#    plt.xlabel('Learning Rate (Log Base 2)', fontsize=18, fontweight='bold')
#    plt.ylabel(ylabel_text, fontsize=18, fontweight='bold')
#    plt.xticks(lrs, lrs, fontsize=12) # Pass array twice to ensure ticks match exact learning rate values
#    plt.yticks(fontsize=12)
#    plt.grid(True, linestyle='--', alpha=0.5)
#    
#    # Legend formatting
#    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
#    for line in leg.get_lines():
#        line.set_marker("") 
#    
#    # --- ADJUSTED AXIS RANGE ---
#    if ylim_range:
#        plt.ylim(ylim_range)
#    
#    # Border styling (Keeping the outer box border intact)
#    ax = plt.gca()
#    for spine in ax.spines.values():
#        spine.set_linewidth(2.0)
#    
#    plt.tight_layout()
#    plt.show()
#
## --- GENERATE PLOTS WITH NEW RANGES ---
#
## 1. Train Loss (Range: 1.15 - 1.70)
#plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.15, 1.70), legend_loc='upper right')
#
## 2. Test Loss (Range: 1.15 - 1.70)
#plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.15, 1.70), legend_loc='upper right')
#
## 3. Valid Loss (Range: 1.15 - 1.70)
#plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', ylim_range=(1.15, 1.70), legend_loc='upper right')
#
## 4. Train Accuracy (Range: 0.35 - 0.58)
#plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')
#
## 5. Test Accuracy (Range: 0.35 - 0.58)
#plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')
#
## 6. Valid Accuracy (Range: 0.35 - 0.58)
#plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', ylim_range=(0.35, 0.58), legend_loc='lower right')



#import matplotlib.pyplot as plt
#import numpy as np
#
## --- SHARED X-AXIS ---
#lrs = np.array([0.100, 0.146, 0.213, 0.311, 0.453, 0.662, 0.965, 1.409, 2.056])
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.7684, 1.6695, 1.6016, 1.5295, 1.4602, 1.4099, 1.3808, 1.3534, 1.3585]
#test_loss_3  = [1.7550, 1.6223, 1.5591, 1.4999, 1.4247, 1.3798, 1.3340, 1.3227, 1.3515]
#valid_loss_3 = [1.7550, 1.6223, 1.5538, 1.4951, 1.4128, 1.3682, 1.3260, 1.3133, 1.3342]
#
#train_acc_3  = [0.3366, 0.3721, 0.4035, 0.4362, 0.4605, 0.4859, 0.4987, 0.5087, 0.5055]
#test_acc_3   = [0.3416, 0.3950, 0.4263, 0.4482, 0.4783, 0.5009, 0.5113, 0.5183, 0.5122]
#valid_acc_3  = [0.3500, 0.4002, 0.4279, 0.4521, 0.4859, 0.5050, 0.5138, 0.5274, 0.5210]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.7000, 1.6326, 1.5342, 1.4770, 1.4384, 1.3768, 1.3668, 1.3305, 1.3219]
#test_loss_6  = [1.6647, 1.5951, 1.4958, 1.4264, 1.3795, 1.3241, 1.3023, 1.3013, 1.2952]
#valid_loss_6 = [1.6644, 1.5969, 1.4960, 1.4253, 1.3725, 1.3236, 1.2912, 1.2760, 1.2794]
#
#train_acc_6  = [0.3510, 0.3767, 0.4249, 0.4514, 0.4684, 0.4919, 0.4992, 0.5159, 0.5177]
#test_acc_6   = [0.3637, 0.3888, 0.4464, 0.4763, 0.4996, 0.5153, 0.5285, 0.5307, 0.5315]
#valid_acc_6  = [0.3664, 0.3928, 0.4479, 0.4814, 0.4942, 0.5140, 0.5286, 0.5391, 0.5290]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.6779, 1.6153, 1.5460, 1.4939, 1.4372, 1.4091, 1.4383, 1.3607, 1.3622]
#test_loss_9  = [1.6592, 1.5717, 1.5335, 1.4675, 1.4240, 1.3836, 1.3674, 1.3176, 1.3340]
#valid_loss_9 = [1.6560, 1.5754, 1.5183, 1.4668, 1.4069, 1.3693, 1.3491, 1.3011, 1.3200]
#
#train_acc_9  = [0.3569, 0.3850, 0.4146, 0.4479, 0.4746, 0.4914, 0.4741, 0.4992, 0.5001]
#test_acc_9   = [0.3723, 0.4075, 0.4261, 0.4622, 0.4813, 0.4941, 0.5040, 0.5176, 0.5150]
#valid_acc_9  = [0.3724, 0.4037, 0.4295, 0.4650, 0.4825, 0.5102, 0.5102, 0.5297, 0.5232]
#
#
#def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
#    plt.figure(figsize=(10, 6))
#    
#    # Plot depths with reduced thickness and dot size
#    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=1.0, label='Depth=3')
#    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=1.0, label='Depth=6')
#    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=1.0, label='Depth=9')
#    
#    # Apply Base-2 Logarithmic Scale to X-Axis
#    plt.xscale('log', base=2)
#    
#    # Labeling
#    plt.xlabel('Learning Rate (Log Base 2)', fontsize=18, fontweight='bold')
#    plt.ylabel(ylabel_text, fontsize=18, fontweight='bold')
#    plt.xticks(lrs, lrs, fontsize=12) # Pass array twice to ensure ticks match exact learning rate values
#    plt.yticks(fontsize=12)
#    plt.grid(True, linestyle='--', alpha=0.5)
#    
#    # Legend formatting
#    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
#    for line in leg.get_lines():
#        line.set_marker("") 
#    
#    # --- ADJUSTED AXIS RANGE ---
#    if ylim_range:
#        plt.ylim(ylim_range)
#    
#    # Border styling (Keeping the outer box border intact)
#    ax = plt.gca()
#    for spine in ax.spines.values():
#        spine.set_linewidth(2.0)
#    
#    plt.tight_layout()
#    plt.show()
#
## --- GENERATE PLOTS WITH NEW RANGES ---
#
## 1. Train Loss (Range: 1.25 - 1.80)
#plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.15, 1.80), legend_loc='upper right')
#
## 2. Test Loss (Range: 1.25 - 1.80)
#plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.15, 1.80), legend_loc='upper right')
#
## 3. Valid Loss (Range: 1.25 - 1.80)
#plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', ylim_range=(1.15, 1.80), legend_loc='upper right')
#
## 4. Train Accuracy (Range: 0.32 - 0.56)
#plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.32, 0.56), legend_loc='lower right')
#
## 5. Test Accuracy (Range: 0.32 - 0.56)
#plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.32, 0.56), legend_loc='lower right')
#
## 6. Valid Accuracy (Range: 0.32 - 0.56)
#plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', ylim_range=(0.32, 0.56), legend_loc='lower right')


#import matplotlib.pyplot as plt
#import numpy as np
#
## --- SHARED X-AXIS ---
#lrs = np.array([0.1, 0.1459, 0.2129, 0.3107, 0.4534, 0.6616, 0.9655, 1.4089, 2.0559])
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.7370, 1.6480, 1.5818, 1.5349, 1.4908, 1.4288, 1.4031, 1.3838, 1.3948]
#test_loss_3  = [1.7397, 1.6453, 1.5845, 1.5292, 1.4722, 1.4174, 1.4155, 1.3821, 1.3910]
#valid_loss_3 = [1.7367, 1.6438, 1.5842, 1.5275, 1.4612, 1.3961, 1.3831, 1.3489, 1.3612]
#
#train_acc_3  = [0.3478, 0.3836, 0.4081, 0.4284, 0.4527, 0.4773, 0.4903, 0.4941, 0.4896]
#test_acc_3   = [0.3491, 0.3889, 0.4141, 0.4361, 0.4680, 0.4861, 0.4903, 0.5014, 0.4965]
#valid_acc_3  = [0.3548, 0.3923, 0.4112, 0.4348, 0.4680, 0.4896, 0.5003, 0.5098, 0.5068]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.7354, 1.6557, 1.6064, 1.5864, 1.5420, 1.4878, 1.4552, 1.3905, 1.3640]
#test_loss_6  = [1.7366, 1.6405, 1.5773, 1.5389, 1.4980, 1.4516, 1.4252, 1.3735, 1.3675]
#valid_loss_6 = [1.7386, 1.6394, 1.5770, 1.5402, 1.4941, 1.4397, 1.4138, 1.3581, 1.3486]
#
#train_acc_6  = [0.3447, 0.3757, 0.3945, 0.4075, 0.4253, 0.4488, 0.4667, 0.4915, 0.5036]
#test_acc_6   = [0.3492, 0.3851, 0.4142, 0.4337, 0.4498, 0.4684, 0.4838, 0.4992, 0.5071]
#valid_acc_6  = [0.3578, 0.3903, 0.4124, 0.4259, 0.4448, 0.4714, 0.4846, 0.5072, 0.5150]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.7576, 1.6904, 1.6155, 1.5600, 1.4851, 1.4826, 1.3947, 1.3585, 1.3519]
#test_loss_9  = [1.7701, 1.6783, 1.6018, 1.5387, 1.4634, 1.4829, 1.4047, 1.3727, 1.3651]
#valid_loss_9 = [1.7692, 1.6748, 1.5928, 1.5301, 1.4510, 1.4697, 1.3839, 1.3487, 1.3425]
#
#train_acc_9  = [0.3325, 0.3600, 0.3904, 0.4178, 0.4479, 0.4572, 0.4906, 0.5021, 0.5018]
#test_acc_9   = [0.3307, 0.3683, 0.4022, 0.4338, 0.4613, 0.4680, 0.4950, 0.5042, 0.5047]
#valid_acc_9  = [0.3367, 0.3725, 0.4070, 0.4313, 0.4602, 0.4671, 0.4953, 0.5099, 0.5094]
#
#
#def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
#    plt.figure(figsize=(10, 6))
#    
#    # Plot depths with thinner lines (linewidth=0.5)
#    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=0.5, label='Depth=3')
#    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=0.5, label='Depth=6')
#    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=0.5, label='Depth=9')
#    
#    # Apply Base-2 Logarithmic Scale to X-Axis
#    plt.xscale('log', base=2)
#    
#    # Labeling - Bigger, thicker text and no "(LR)"
#    plt.xlabel('Learning Rate', fontsize=22, fontweight='black')
#    plt.ylabel(ylabel_text, fontsize=22, fontweight='black')
#    
#    plt.xticks(lrs, [f"{x:.4f}".rstrip('0').rstrip('.') for x in lrs], fontsize=12) 
#    plt.yticks(fontsize=12)
#    plt.grid(True, linestyle='--', alpha=0.5)
#    
#    # Legend formatting
#    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
#    for line in leg.get_lines():
#        line.set_marker("") 
#    
#    # --- ADJUSTED AXIS RANGE ---
#    if ylim_range:
#        plt.ylim(ylim_range)
#    
#    # Border styling
#    ax = plt.gca()
#    for spine in ax.spines.values():
#        spine.set_linewidth(2.0)
#    
#    plt.tight_layout()
#    plt.show()
#
## --- GENERATE PLOTS WITH NEW RANGES ---
#
## 1. Train Loss
#plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.15, 1.80), legend_loc='upper right')
#
## 2. Test Loss 
#plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.15, 1.80), legend_loc='upper right')
#
## 3. Valid Loss
#plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', ylim_range=(1.15, 1.80), legend_loc='upper right')
#
## 4. Train Accuracy
#plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.32, 0.56), legend_loc='lower right')
#
## 5. Test Accuracy
#plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.32, 0.56), legend_loc='lower right')
#
## 6. Valid Accuracy
#plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', ylim_range=(0.32, 0.56), legend_loc='lower right')


#import matplotlib.pyplot as plt
#import numpy as np
#
## --- SHARED X-AXIS ---
#lrs = np.array([0.1, 0.1459, 0.2129, 0.3107, 0.4534, 0.6616, 0.9655, 1.4089, 2.0559])
#
## --- DATA FOR DEPTH 3 (Orange) ---
#train_loss_3 = [1.8721, 1.8016, 1.7325, 1.6430, 1.5932, 1.5389, 1.5073, 1.4906, 1.4795]
#test_loss_3  = [1.8465, 1.7753, 1.6809, 1.5955, 1.5410, 1.5072, 1.4722, 1.4472, 1.4266]
#valid_loss_3 = [1.8523, 1.7789, 1.6786, 1.5889, 1.5328, 1.4963, 1.4581, 1.4336, 1.4184]
#
#train_acc_3  = [0.2983, 0.3261, 0.3540, 0.3927, 0.4136, 0.4354, 0.4466, 0.4513, 0.4577]
#test_acc_3   = [0.3121, 0.3394, 0.3795, 0.4137, 0.4366, 0.4520, 0.4645, 0.4709, 0.4815]
#valid_acc_3  = [0.3141, 0.3409, 0.3828, 0.4168, 0.4386, 0.4577, 0.4709, 0.4782, 0.4863]
#
## --- DATA FOR DEPTH 6 (Red) ---
#train_loss_6 = [1.8333, 1.7584, 1.6643, 1.6005, 1.5419, 1.4874, 1.4428, 1.4141, 1.4324]
#test_loss_6  = [1.8179, 1.7484, 1.6454, 1.5630, 1.4897, 1.4415, 1.4186, 1.3777, 1.3819]
#valid_loss_6 = [1.8248, 1.7497, 1.6405, 1.5554, 1.4860, 1.4360, 1.4060, 1.3669, 1.3729]
#
#train_acc_6  = [0.3042, 0.3321, 0.3693, 0.3975, 0.4254, 0.4493, 0.4675, 0.4830, 0.4735]
#test_acc_6   = [0.3090, 0.3430, 0.3801, 0.4175, 0.4530, 0.4729, 0.4812, 0.4966, 0.4945]
#valid_acc_6  = [0.3116, 0.3439, 0.3834, 0.4161, 0.4515, 0.4722, 0.4877, 0.5027, 0.5017]
#
## --- DATA FOR DEPTH 9 (Purple) ---
#train_loss_9 = [1.8049, 1.7363, 1.6801, 1.5947, 1.5290, 1.5054, 1.5004, 1.4552, 1.4602]
#test_loss_9  = [1.7781, 1.7105, 1.6667, 1.5647, 1.4764, 1.4694, 1.4392, 1.4068, 1.4201]
#valid_loss_9 = [1.7855, 1.7129, 1.6601, 1.5616, 1.4734, 1.4686, 1.4315, 1.4004, 1.4113]
#
#train_acc_9  = [0.3085, 0.3373, 0.3603, 0.4005, 0.4308, 0.4450, 0.4472, 0.4649, 0.4643]
#test_acc_9   = [0.3241, 0.3473, 0.3713, 0.4181, 0.4562, 0.4650, 0.4741, 0.4868, 0.4836]
#valid_acc_9  = [0.3249, 0.3455, 0.3750, 0.4201, 0.4583, 0.4619, 0.4808, 0.4891, 0.4888]
#
#
#def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
#    plt.figure(figsize=(10, 6))
#    
#    # Plot depths with thinner lines (linewidth=0.5)
#    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=0.5, label='Depth=3')
#    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=0.5, label='Depth=6')
#    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=0.5, label='Depth=9')
#    
#    # Apply Base-2 Logarithmic Scale to X-Axis
#    plt.xscale('log', base=2)
#    
#    # Labeling - Bigger, thicker text and no "(LR)"
#    plt.xlabel('Learning Rate', fontsize=22, fontweight='black')
#    plt.ylabel(ylabel_text, fontsize=22, fontweight='black')
#    
#    plt.xticks(lrs, [f"{x:.4f}".rstrip('0').rstrip('.') for x in lrs], fontsize=12) 
#    plt.yticks(fontsize=12)
#    plt.grid(True, linestyle='--', alpha=0.5)
#    
#    # Legend formatting
#    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
#    for line in leg.get_lines():
#        line.set_marker("") 
#    
#    # --- ADJUSTED AXIS RANGE ---
#    if ylim_range:
#        plt.ylim(ylim_range)
#    
#    # Border styling
#    ax = plt.gca()
#    for spine in ax.spines.values():
#        spine.set_linewidth(2.0)
#    
#    plt.tight_layout()
#    plt.show()
#
## --- GENERATE PLOTS WITH NEW RANGES ---
#
## 1. Train Loss
#plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', ylim_range=(1.35, 1.90), legend_loc='upper right')
#
## 2. Test Loss 
#plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', ylim_range=(1.35, 1.90), legend_loc='upper right')
#
## 3. Valid Loss
#plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', ylim_range=(1.35, 1.90), legend_loc='upper right')
#
## 4. Train Accuracy
#plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', ylim_range=(0.28, 0.52), legend_loc='lower right')
#
## 5. Test Accuracy
#plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', ylim_range=(0.28, 0.52), legend_loc='lower right')
#
## 6. Valid Accuracy
#plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', ylim_range=(0.28, 0.52), legend_loc='lower right')






import matplotlib.pyplot as plt
import numpy as np

# --- SHARED X-AXIS ---
lrs = np.array([0.1, 0.1459, 0.2129, 0.3107, 0.4534, 0.6616, 0.9655, 1.4089, 2.0559])

# --- DATA FOR DEPTH 3 (Orange) ---
train_loss_3 = [1.8721, 1.8016, 1.7325, 1.6430, 1.5932, 1.5389, 1.5073, 1.4906, 1.4795]
valid_loss_3 = [1.8523, 1.7789, 1.6786, 1.5889, 1.5328, 1.4963, 1.4581, 1.4336, 1.4184]
test_loss_3  = [1.8465, 1.7753, 1.6809, 1.5955, 1.5410, 1.5072, 1.4722, 1.4472, 1.4266]

train_acc_3  = [0.2983, 0.3261, 0.3540, 0.3927, 0.4136, 0.4354, 0.4466, 0.4513, 0.4577]
valid_acc_3  = [0.3141, 0.3409, 0.3828, 0.4168, 0.4386, 0.4577, 0.4709, 0.4782, 0.4863]
test_acc_3   = [0.3121, 0.3394, 0.3795, 0.4137, 0.4366, 0.4520, 0.4645, 0.4709, 0.4815]

# --- DATA FOR DEPTH 6 (Red) ---
train_loss_6 = [1.8333, 1.7584, 1.6643, 1.6005, 1.5419, 1.4874, 1.4428, 1.4141, 1.4324]
valid_loss_6 = [1.8248, 1.7497, 1.6405, 1.5554, 1.4860, 1.4360, 1.4060, 1.3669, 1.3729]
test_loss_6  = [1.8179, 1.7484, 1.6454, 1.5630, 1.4897, 1.4415, 1.4186, 1.3777, 1.3819]

train_acc_6  = [0.3042, 0.3321, 0.3693, 0.3975, 0.4254, 0.4493, 0.4675, 0.4830, 0.4735]
valid_acc_6  = [0.3116, 0.3439, 0.3834, 0.4161, 0.4515, 0.4722, 0.4877, 0.5027, 0.5017]
test_acc_6   = [0.3090, 0.3430, 0.3801, 0.4175, 0.4530, 0.4729, 0.4812, 0.4966, 0.4945]

# --- DATA FOR DEPTH 9 (Purple) ---
train_loss_9 = [1.8049, 1.7363, 1.6801, 1.5947, 1.5290, 1.5054, 1.5004, 1.4552, 1.4602]
valid_loss_9 = [1.7855, 1.7129, 1.6601, 1.5616, 1.4734, 1.4686, 1.4315, 1.4004, 1.4113]
test_loss_9  = [1.7781, 1.7105, 1.6667, 1.5647, 1.4764, 1.4694, 1.4392, 1.4068, 1.4201]

train_acc_9  = [0.3085, 0.3373, 0.3603, 0.4005, 0.4308, 0.4450, 0.4472, 0.4649, 0.4643]
valid_acc_9  = [0.3249, 0.3455, 0.3750, 0.4201, 0.4583, 0.4619, 0.4808, 0.4891, 0.4888]
test_acc_9   = [0.3241, 0.3473, 0.3713, 0.4181, 0.4562, 0.4650, 0.4741, 0.4868, 0.4836]

def plot_metric(y1, y2, y3, ylabel_text, ylim_range, legend_loc='best'):
    plt.figure(figsize=(10, 6))
    
    # Plot depths with thinner lines (linewidth=0.5)
    plt.plot(lrs, y1, color='orange', marker='o', markersize=4, linewidth=0.5, label='Depth=3')
    plt.plot(lrs, y2, color='red', marker='o', markersize=4, linewidth=0.5, label='Depth=6')
    plt.plot(lrs, y3, color='purple', marker='o', markersize=4, linewidth=0.5, label='Depth=9')
    
    # Apply Base-2 Logarithmic Scale to X-Axis
    plt.xscale('log', base=2)
    
    # Labeling
    plt.xlabel('Learning Rate', fontsize=22, fontweight='black')
    plt.ylabel(ylabel_text, fontsize=22, fontweight='black')
    
    plt.xticks(lrs, [f"{x:.4f}".rstrip('0').rstrip('.') for x in lrs], fontsize=12) 
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Legend formatting
    leg = plt.legend(fontsize=14, loc=legend_loc, frameon=True, edgecolor='black')
    for line in leg.get_lines():
        line.set_marker("") 
    
    # Adjust axis range
    if ylim_range:
        plt.ylim(ylim_range)
    
    # Border styling
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    plt.tight_layout()
    plt.show()

# --- GENERATE PLOTS ---

# Loss Plots
plot_metric(train_loss_3, train_loss_6, train_loss_9, 'Train Loss', (1.35, 1.90), 'upper right')
plot_metric(test_loss_3, test_loss_6, test_loss_9, 'Test Loss', (1.35, 1.90), 'upper right')
plot_metric(valid_loss_3, valid_loss_6, valid_loss_9, 'Valid Loss', (1.35, 1.90), 'upper right')

# Accuracy Plots
plot_metric(train_acc_3, train_acc_6, train_acc_9, 'Train Accuracy', (0.28, 0.52), 'lower right')
plot_metric(test_acc_3, test_acc_6, test_acc_9, 'Test Accuracy', (0.28, 0.52), 'lower right')
plot_metric(valid_acc_3, valid_acc_6, valid_acc_9, 'Valid Accuracy', (0.28, 0.52), 'lower right')