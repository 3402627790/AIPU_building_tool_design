import torch
import numpy as np


def fma(a, b, c):
    def to_np(a):
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        return a
    import ctypes
    import platform
    libm = ctypes.CDLL("libm.so.6" if platform.system() == "Linux" else None)
    if libm:
        fmaf_func = libm.fmaf
        fmaf_func.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        fmaf_func.restype = ctypes.c_float
    fma = np.frompyfunc(lambda x, y, z: fmaf_func(x, y, z), 3, 1)
    a = to_np(a)
    b = to_np(b)
    c = to_np(c)
    return fma(a, b, c).astype("float32")


inverse_sqrt_table = [32767, 32515, 32268, 32026, 31790, 31558, 31332, 31111, 30894,
                      30682, 30474, 30270, 30070, 29874, 29682, 29494, 29309, 29127,
                      28949, 28774, 28602, 28434, 28268, 28105, 27945, 27787, 27632,
                      27480, 27330, 27183, 27038, 26895, 26755, 26617, 26481, 26346,
                      26214, 26084, 25956, 25830, 25705, 25583, 25462, 25342, 25225,
                      25109, 24994, 24882, 24770, 24660, 24552, 24445, 24339, 24235,
                      24132, 24031, 23930, 23831, 23733, 23637, 23541, 23447, 23354,
                      23262, 23170, 23080, 22992, 22904, 22817, 22731, 22646, 22562,
                      22479, 22396, 22315, 22235, 22155, 22077, 21999, 21922, 21845,
                      21770, 21695, 21621, 21548, 21476, 21404, 21333, 21263, 21193,
                      21124, 21056, 20988, 20921, 20855, 20789, 20724, 20660, 20596,
                      20533, 20470, 20408, 20346, 20285, 20225, 20165, 20106, 20047,
                      19988, 19930, 19873, 19816, 19760, 19704, 19649, 19594, 19539,
                      19485, 19431, 19378, 19326, 19273, 19221, 19170, 19119, 19068,
                      19018, 18968, 18919, 18870, 18821, 18773, 18725, 18677, 18630,
                      18583, 18536, 18490, 18444, 18399, 18354, 18309, 18264, 18220,
                      18176, 18133, 18090, 18047, 18004, 17962, 17920, 17878, 17837,
                      17795, 17755, 17714, 17674, 17634, 17594, 17554, 17515, 17476,
                      17438, 17399, 17361, 17323, 17285, 17248, 17211, 17174, 17137,
                      17100, 17064, 17028, 16992, 16957, 16921, 16886, 16851, 16817,
                      16782, 16748, 16714, 16680, 16646, 16613, 16579, 16546, 16514,
                      16481, 16448, 16416, 16384]

# 32bit  data
table_Log2_q15 = [0, 184, 368, 551, 733, 914, 1095, 1275, 1455, 1633,
                  1811, 1989, 2166, 2342, 2517, 2692, 2866, 3039, 3212, 3385,
                  3556, 3727, 3897, 4067, 4236, 4405, 4573, 4740, 4907, 5073,
                  5239, 5404, 5568, 5732, 5895, 6058, 6220, 6382, 6543, 6703,
                  6863, 7023, 7182, 7340, 7498, 7655, 7812, 7968, 8124, 8279,
                  8434, 8588, 8742, 8895, 9048, 9200, 9352, 9503, 9654, 9804,
                  9954, 10104, 10253, 10401, 10549, 10696, 10843, 10990, 11136, 11282,
                  11427, 11572, 11716, 11860, 12004, 12147, 12289, 12431, 12573, 12715,
                  12855, 12996, 13136, 13276, 13415, 13554, 13692, 13830, 13968, 14105,
                  14242, 14378, 14514, 14650, 14785, 14920, 15055, 15189, 15322, 15456,
                  15589, 15721, 15854, 15986, 16117, 16248, 16379, 16509, 16639, 16769,
                  16898, 17027, 17156, 17284, 17412, 17540, 17667, 17794, 17921, 18047,
                  18173, 18298, 18424, 18548, 18673, 18797, 18921, 19045, 19168, 19291,
                  19414, 19536, 19658, 19780, 19901, 20022, 20143, 20263, 20383, 20503,
                  20623, 20742, 20861, 20980, 21098, 21216, 21334, 21451, 21568, 21685,
                  21802, 21918, 22034, 22150, 22265, 22380, 22495, 22610, 22724, 22838,
                  22952, 23066, 23179, 23292, 23404, 23517, 23629, 23741, 23852, 23964,
                  24075, 24186, 24296, 24407, 24517, 24627, 24736, 24845, 24955, 25063,
                  25172, 25280, 25388, 25496, 25604, 25711, 25818, 25925, 26031, 26138,
                  26244, 26350, 26455, 26561, 26666, 26771, 26876, 26980, 27084, 27188,
                  27292, 27396, 27499, 27602, 27705, 27808, 27910, 28012, 28114, 28216,
                  28318, 28419, 28520, 28621, 28722, 28822, 28922, 29022, 29122, 29222,
                  29321, 29421, 29520, 29618, 29717, 29815, 29914, 30012, 30109, 30207,
                  30304, 30401, 30498, 30595, 30692, 30788, 30884, 30980, 31076, 31172,
                  31267, 31362, 31457, 31552, 31647, 31741, 31836, 31930, 32024, 32117,
                  32211, 32304, 32397, 32490, 32583, 32676]

# table_Log2_Q31 =[          0,    95335645,   187825021,   277633165,   364911162,
#          449797678,   532420281,   612896598,   691335320,   767837083,
#          842495250,   915396590,   986621888,  1056246482,  1124340739,
#         1190970490,  1256197405,  1320079339,  1382670639,  1444022426,
#         1504182841,  1563197273,  1621108567,  1677957208,  1733781493,
#         1788617686,  1842500157,  1895461516,  1947532725,  1998743213,
#         2049120974,  2098692655, 2147483647]


table_pow2_q15 = [
    32768, 32857, 32946, 33035, 33125, 33215, 33305, 33395, 33486, 33576,
    33667, 33759, 33850, 33942, 34034, 34126, 34219, 34312, 34405, 34498,
    34591, 34685, 34779, 34874, 34968, 35063, 35158, 35253, 35349, 35445,
    35541, 35637, 35734, 35831, 35928, 36025, 36123, 36221, 36319, 36417,
    36516, 36615, 36715, 36814, 36914, 37014, 37114, 37215, 37316, 37417,
    37518, 37620, 37722, 37824, 37927, 38030, 38133, 38236, 38340, 38444,
    38548, 38653, 38757, 38863, 38968, 39074, 39180, 39286, 39392, 39499,
    39606, 39714, 39821, 39929, 40037, 40146, 40255, 40364, 40473, 40583,
    40693, 40804, 40914, 41025, 41136, 41248, 41360, 41472, 41584, 41697,
    41810, 41923, 42037, 42151, 42265, 42380, 42495, 42610, 42726, 42841,
    42958, 43074, 43191, 43308, 43425, 43543, 43661, 43780, 43898, 44017,
    44137, 44256, 44376, 44497, 44617, 44738, 44859, 44981, 45103, 45225,
    45348, 45471, 45594, 45718, 45842, 45966, 46091, 46216, 46341, 46467,
    46593, 46719, 46846, 46973, 47100, 47228, 47356, 47484, 47613, 47742,
    47871, 48001, 48131, 48262, 48393, 48524, 48655, 48787, 48920, 49052,
    49185, 49319, 49452, 49586, 49721, 49856, 49991, 50126, 50262, 50399,
    50535, 50672, 50810, 50947, 51085, 51224, 51363, 51502, 51642, 51782,
    51922, 52063, 52204, 52346, 52488, 52630, 52773, 52916, 53059, 53203,
    53347, 53492, 53637, 53782, 53928, 54074, 54221, 54368, 54515, 54663,
    54811, 54960, 55109, 55258, 55408, 55558, 55709, 55860, 56012, 56163,
    56316, 56468, 56622, 56775, 56929, 57083, 57238, 57393, 57549, 57705,
    57861, 58018, 58176, 58333, 58491, 58650, 58809, 58968, 59128, 59289,
    59449, 59611, 59772, 59934, 60097, 60260, 60423, 60587, 60751, 60916,
    61081, 61247, 61413, 61579, 61746, 61914, 62081, 62250, 62419, 62588,
    62757, 62928, 63098, 63269, 63441, 63613, 63785, 63958, 64132, 64306,
    64480, 64655, 64830, 65006, 65182, 65359
]


def Log2_norm(tx, loginQ, logoutQ):
    vshape = tx.shape
    x = tx.reshape(-1)
    var_out = torch.zeros_like(x)
    # search counting the leading zero of the value with asm instruction(clz)
    for k in range(0, 30):
        cond = x < 0x40000000
        var_out = torch.where(cond, var_out + 1, var_out)
        x = torch.where(cond, x << 1, x)

    x = ((x) >> 22) & 0xff
    # x = (x>> 9)
    # frac =  (x&0x7fff)
    table_Log2_norm = torch.tensor(
        table_Log2_q15, dtype=torch.int64, device=x.device)
    y = torch.gather(table_Log2_norm, 0, x.reshape(-1))
    del table_Log2_norm
    y = y << 15
    y = (y) >> (30 - logoutQ)
    # tmp1 = torch.gather(table_Log2_norm, 0, (i+1).reshape(-1))
    # diff = (tmp1-tmp0)>>15
    # y =  (tmp0+(torch.multiply(diff,frac)))>>(31-logoutQ)
    y = y - ((var_out - (30 - loginQ)) << logoutQ)
    del var_out
    return y.reshape(vshape)


def simple_log2(tx, Pow2OutQ, adjust_q):
    vshape = tx.shape
    x = tx.reshape(-1)
    var_out = torch.zeros_like(x)
    # search counting the leading zero of the value with asm instruction(clz)
    for k in range(0, 30):
        cond = x < 0x40000000
        var_out = torch.where(cond, var_out + 1, var_out)
        x = torch.where(cond, x << 1, x)
    if adjust_q == 0:
        y = torch.zeros_like(x)
    else:
        y = (x > 1518500250).int()
    y = y - ((var_out - (30 - Pow2OutQ)) << adjust_q)

    return y.reshape(vshape)
    # input tx is 16bit negative data, pow2inQ


def Pow2(tx, adjust_q, Pow2OutQ):
    vshape = tx.shape
    x = tx.reshape(-1)

    exponant = (x >> adjust_q).to(torch.int32)
    x = (x & ((1 << adjust_q) - 1)).long() >> (adjust_q - 8)

    table_pow2 = torch.tensor(
        table_pow2_q15, dtype=torch.int32, device=x.device)
    x = torch.gather(table_pow2, 0, x.reshape(-1))
    del table_pow2

    x = x << 15
    x = x >> (30 - Pow2OutQ - exponant)
    del exponant

    return x.view(vshape)


def broadcasting_transform(x0, x1):
    from AIPUBuilder.logger import WARN
    # align axis
    x0_shape, x1_shape = x0.shape, x1.shape
    max_len = max(len(x0_shape), len(x1_shape))
    tile_a, tile_b = [1] * max_len, [1] * max_len
    x0, x1 = x0.tile(tile_a), x1.tile(tile_b)

    # check broadcast params
    for i in range(max_len):
        local_axis = max_len - i - 1
        if x0.shape[local_axis] == x1.shape[local_axis]:
            pass
        elif x0.shape[local_axis] == 1 and x1.shape[local_axis] % x0.shape[local_axis] == 0:
            tile_a[local_axis] = x1.shape[local_axis]
        elif x1.shape[local_axis] == 1 and x0.shape[i] % x1.shape[local_axis] == 0:
            tile_b[local_axis] = x0.shape[local_axis]
        elif x1.shape[local_axis] % x0.shape[local_axis] == 0:
            tile_a[local_axis] = int(
                x1.shape[local_axis] / x0.shape[local_axis])
        elif x0.shape[local_axis] % x1.shape[local_axis] == 0:
            tile_b[local_axis] = int(
                x0.shape[local_axis] / x1.shape[local_axis])
        elif x1.shape[local_axis] % x0.shape[local_axis] != 0 or x0.shape[local_axis] % x1.shape[local_axis] != 0:
            WARN('tensors can not broadcast')
    x0, x1 = x0.tile(tile_a), x1.tile(tile_b)

    return x0, x1


#################################### for softmax aiff #####################################
# from .hex_fp_util import *
# fp24
# lookup 2**0~2**1, 2**(1/(2**9)) = 1.0013
pow2_f_lut = [1.0000, 1.0013, 1.0027, 1.0041, 1.0054, 1.0068, 1.0081, 1.0095, 1.0109,
              1.0122, 1.0136, 1.0150, 1.0164, 1.0177, 1.0191, 1.0205, 1.0219, 1.0233,
              1.0247, 1.0260, 1.0274, 1.0288, 1.0302, 1.0316, 1.0330, 1.0344, 1.0358,
              1.0372, 1.0386, 1.0400, 1.0414, 1.0428, 1.0443, 1.0457, 1.0471, 1.0485,
              1.0499, 1.0514, 1.0528, 1.0542, 1.0556, 1.0571, 1.0585, 1.0599, 1.0614,
              1.0628, 1.0642, 1.0657, 1.0671, 1.0686, 1.0700, 1.0715, 1.0729, 1.0744,
              1.0758, 1.0773, 1.0787, 1.0802, 1.0817, 1.0831, 1.0846, 1.0861, 1.0876,
              1.0890, 1.0905, 1.0920, 1.0934, 1.0949, 1.0964, 1.0979, 1.0994, 1.1009,
              1.1024, 1.1039, 1.1053, 1.1068, 1.1084, 1.1099, 1.1114, 1.1129, 1.1144,
              1.1159, 1.1174, 1.1189, 1.1204, 1.1219, 1.1235, 1.1250, 1.1265, 1.1280,
              1.1295, 1.1311, 1.1326, 1.1342, 1.1357, 1.1372, 1.1388, 1.1403, 1.1419,
              1.1434, 1.1450, 1.1465, 1.1481, 1.1496, 1.1512, 1.1527, 1.1543, 1.1559,
              1.1574, 1.1590, 1.1606, 1.1621, 1.1637, 1.1653, 1.1669, 1.1685, 1.1700,
              1.1716, 1.1732, 1.1748, 1.1764, 1.1780, 1.1796, 1.1812, 1.1828, 1.1844,
              1.1860, 1.1876, 1.1892, 1.1908, 1.1924, 1.1940, 1.1956, 1.1973, 1.1989,
              1.2005, 1.2021, 1.2038, 1.2054, 1.2070, 1.2087, 1.2103, 1.2119, 1.2136,
              1.2152, 1.2169, 1.2185, 1.2202, 1.2218, 1.2235, 1.2252, 1.2268, 1.2285,
              1.2301, 1.2318, 1.2335, 1.2351, 1.2368, 1.2385, 1.2402, 1.2419, 1.2435,
              1.2452, 1.2469, 1.2486, 1.2503, 1.2520, 1.2537, 1.2554, 1.2571, 1.2588,
              1.2605, 1.2622, 1.2639, 1.2656, 1.2673, 1.2690, 1.2708, 1.2725, 1.2742,
              1.2759, 1.2776, 1.2794, 1.2811, 1.2829, 1.2846, 1.2863, 1.2881, 1.2898,
              1.2916, 1.2933, 1.2951, 1.2968, 1.2986, 1.3004, 1.3021, 1.3039, 1.3056,
              1.3074, 1.3092, 1.3109, 1.3127, 1.3145, 1.3163, 1.3181, 1.3199, 1.3216,
              1.3234, 1.3252, 1.3270, 1.3288, 1.3306, 1.3324, 1.3342, 1.3360, 1.3378,
              1.3397, 1.3415, 1.3433, 1.3451, 1.3469, 1.3488, 1.3506, 1.3524, 1.3542,
              1.3561, 1.3579, 1.3597, 1.3616, 1.3634, 1.3653, 1.3671, 1.3690, 1.3708,
              1.3727, 1.3746, 1.3764, 1.3783, 1.3802, 1.3820, 1.3839, 1.3858, 1.3876,
              1.3895, 1.3914, 1.3933, 1.3952, 1.3971, 1.3990, 1.4008, 1.4028, 1.4047,
              1.4066, 1.4085, 1.4104, 1.4123, 1.4142, 1.4161, 1.4180, 1.4200, 1.4219,
              1.4238, 1.4257, 1.4277, 1.4296, 1.4315, 1.4335, 1.4354, 1.4373, 1.4393,
              1.4413, 1.4432, 1.4452, 1.4471, 1.4491, 1.4510, 1.4530, 1.4550, 1.4569,
              1.4589, 1.4609, 1.4629, 1.4649, 1.4669, 1.4688, 1.4708, 1.4728, 1.4748,
              1.4768, 1.4788, 1.4808, 1.4828, 1.4848, 1.4868, 1.4889, 1.4909, 1.4929,
              1.4949, 1.4969, 1.4990, 1.5010, 1.5030, 1.5051, 1.5071, 1.5092, 1.5112,
              1.5132, 1.5153, 1.5173, 1.5194, 1.5215, 1.5235, 1.5256, 1.5276, 1.5297,
              1.5318, 1.5339, 1.5359, 1.5380, 1.5401, 1.5422, 1.5443, 1.5464, 1.5485,
              1.5506, 1.5527, 1.5548, 1.5569, 1.5590, 1.5611, 1.5632, 1.5653, 1.5674,
              1.5696, 1.5717, 1.5738, 1.5760, 1.5781, 1.5802, 1.5824, 1.5845, 1.5867,
              1.5888, 1.5910, 1.5931, 1.5953, 1.5974, 1.5996, 1.6018, 1.6039, 1.6061,
              1.6083, 1.6105, 1.6127, 1.6148, 1.6170, 1.6192, 1.6214, 1.6236, 1.6258,
              1.6280, 1.6302, 1.6324, 1.6346, 1.6368, 1.6391, 1.6413, 1.6435, 1.6458,
              1.6480, 1.6502, 1.6524, 1.6547, 1.6569, 1.6591, 1.6614, 1.6637, 1.6659,
              1.6682, 1.6704, 1.6727, 1.6750, 1.6772, 1.6795, 1.6818, 1.6841, 1.6863,
              1.6886, 1.6909, 1.6932, 1.6955, 1.6978, 1.7001, 1.7024, 1.7047, 1.7070,
              1.7093, 1.7116, 1.7140, 1.7163, 1.7186, 1.7209, 1.7233, 1.7256, 1.7279,
              1.7303, 1.7326, 1.7350, 1.7373, 1.7397, 1.7420, 1.7444, 1.7468, 1.7491,
              1.7515, 1.7539, 1.7562, 1.7586, 1.7610, 1.7634, 1.7658, 1.7682, 1.7706,
              1.7729, 1.7754, 1.7778, 1.7802, 1.7826, 1.7850, 1.7874, 1.7899, 1.7923,
              1.7947, 1.7971, 1.7996, 1.8020, 1.8044, 1.8069, 1.8093, 1.8118, 1.8142,
              1.8167, 1.8192, 1.8216, 1.8241, 1.8266, 1.8290, 1.8315, 1.8340, 1.8365,
              1.8390, 1.8415, 1.8440, 1.8464, 1.8489, 1.8515, 1.8540, 1.8565, 1.8590,
              1.8615, 1.8640, 1.8665, 1.8691, 1.8716, 1.8741, 1.8767, 1.8792, 1.8818,
              1.8843, 1.8869, 1.8894, 1.8920, 1.8946, 1.8971, 1.8997, 1.9023, 1.9048,
              1.9074, 1.9100, 1.9126, 1.9152, 1.9178, 1.9204, 1.9230, 1.9256, 1.9282,
              1.9308, 1.9334, 1.9360, 1.9387, 1.9413, 1.9439, 1.9466, 1.9492, 1.9518,
              1.9545, 1.9571, 1.9598, 1.9624, 1.9651, 1.9677, 1.9704, 1.9731, 1.9758,
              1.9785, 1.9811, 1.9838, 1.9865, 1.9892, 1.9919, 1.9946, 1.9973, 2.0000]

########################################################################################
# slimulate hardware behaviour to implement float exp by LUT
########################################################################################

# torch.set_printoptions(precision=8)


def fp32_to_fp24(fp32_data, is_trunc=False):
    is_scalar = False
    if isinstance(fp32_data, float):
        is_scalar = True
        fp32_data = torch.Tensor([fp32_data]).to(torch.float32)
    int_v = (fp32_data.float()).view(torch.int)
    fp24_data = (int_v & 0xffffff00)
    if not is_trunc:
        low_8 = int_v & 0xff
        tmp_v = int_v & 0x100
        is_carry_bit = torch.logical_or(low_8 > 0x80, torch.logical_and(
            low_8 == 0x80, tmp_v == 0x100))  # round to nearest even
        fp24_data[is_carry_bit] += 0x100
    fp24_data = fp24_data.view(torch.float32)
    if is_scalar:
        fp24_data = fp24_data.item()
    return fp24_data


# calc 2^x, align to grm aiff
def fexp(f_vdata):
    posinf_mask = np.isposinf(f_vdata.cpu().numpy())
    neginf_mask = np.isneginf(f_vdata.cpu().numpy())
    nan_mask = np.isnan(f_vdata.cpu().numpy())
    unexpected_mask = posinf_mask | neginf_mask | nan_mask
    f_vdata[unexpected_mask] = 0  # unexpected data will raise error

    mantisa_bit = 16
    lut_bits = 9
    vshape = f_vdata.shape

    # limit input,  clip to [-126.9, 126.9]
    f_vdata = (f_vdata < -126.9) * -126.9 + f_vdata * (f_vdata >= -126.9)
    f_vdata = (f_vdata > 126.9) * 126.9 + f_vdata * (f_vdata < 126.9)

    exponent = f_vdata.floor()
    mantisa = ((f_vdata - exponent) * ((1 << mantisa_bit) - 1)).long()
    index = mantisa >> 7  # high 9 bit, 2**9=512, len(lut)=513

    interp_bit = (mantisa & 0x7f) << 9  # low 7 bit, <<9 is align to grm

    torch_pow2_f_lut = torch.tensor(pow2_f_lut).to(f_vdata.device)
    diff = torch_pow2_f_lut[index + 1] - torch_pow2_f_lut[index]
    diff = fp32_to_fp24(diff)

    # lut[0] + (lut[1]-lut[0])*interp
    # 2^(exponent+mantisa) = 2^exponent * 2^(mantisa)
    post_lut = (diff * interp_bit) * (2.0 ** -(7 + 9))
    post_lut = fp32_to_fp24(post_lut)
    pow2_factor = (torch_pow2_f_lut[index] + post_lut) * (2.0 ** exponent)
    pow2_fp24 = fp32_to_fp24(pow2_factor)
    # pow2_fp24 = ((pow2_factor).view(torch.int) & 0xffffFF00).view(torch.float32)

    pow2_fp24[posinf_mask] = torch.inf  # exp(inf) = inf
    pow2_fp24[neginf_mask] = 0          # exp(-inf) = 0
    pow2_fp24[nan_mask] = 1             # exp(nan) = 1 in grm
    return pow2_fp24.reshape(vshape)


# rsqrt fp align to grm
def f32_rsqrt(fp32_data):
    is_ndarray = True if isinstance(fp32_data, np.ndarray) else False
    fp32_data = torch.from_numpy(fp32_data).to(
        torch.float32) if is_ndarray else fp32_data
    int_v = (fp32_data).view(torch.int)
    # (v < nan/inf) or (v == -inf) or (v < 0 and v != -0)
    # torch.IntTensor([0xFF800000]) overflow in torch 2.5
    neg_inf = torch.asarray(-torch.inf, device=int_v.device).view(torch.int)
    # torch.IntTensor([0x80000000])
    neg_zero = (-torch.zeros(1, device=int_v.device)).view(torch.int)
    flag1 = torch.logical_or((int_v & torch.tensor(
        [0x7fc00000], dtype=torch.int32, device=int_v.device) == 0), (int_v == neg_inf))
    flag2 = torch.logical_and((int_v & neg_zero), (int_v != neg_zero))
    flag = torch.logical_or(flag1, flag2).cpu().numpy()
    fp32_data = fp32_data.cpu().numpy()
    # sqrt(1/x), torch.sqrt has 1 ulp rounding error -> np.sqrt
    out = np.sqrt(1 / fp32_data)
    out[flag] = 1 / (np.sqrt(fp32_data[flag]))  # 1 / sqrt(x)
    out = torch.from_numpy(out).to(int_v.device) if not is_ndarray else out
    return out


# return transpose perm to place the axis dimension last, for normalization/softmax...
def axis_to_perm(axis, dim):
    axis = [axis] if isinstance(axis, int) else axis
    perm = []
    for i in range(dim):
        if i not in axis:
            perm.append(i)
    perm += axis
    return perm


# used after axis_to_perm, get perm to restore origin shape
def invert_perm(front_perm):
    dim = len(front_perm)
    back_perm = front_perm.copy()
    for i in range(dim):
        where_idx_from = front_perm[i]
        back_perm[where_idx_from] = i
    return back_perm


# align to lib, calc 8 data at a time, 16xfp16 -> 8xfp32_e,8xfp32_o, 1st_8_e+1st_8_o+2nd_8_e+2nd_8_o+...
def sum_square_fp16(data1, tmp_len, data_tmp, data_tmp_square=[], tec_num=1, merged=False):
    # if merged=True, (1st_8_e+1st_8_o)+(2nd_8_e+2nd_8_o)+...
    def sum_fp16_lib(data_tmp1, data_tmp, data_tmp_square):
        if merged:
            data_tmp += (data_tmp1[..., 0::2] + data_tmp1[..., 1::2])
        else:
            data_tmp += data_tmp1[..., 0::2]
            data_tmp += data_tmp1[..., 1::2]
        if len(data_tmp_square) > 0:
            if merged:
                data_tmp_square += (data_tmp1[..., 0::2]
                                    ** 2 + data_tmp1[..., 1::2] ** 2)
            else:
                data_tmp_square += data_tmp1[..., 0::2] ** 2
                data_tmp_square += data_tmp1[..., 1::2] ** 2
        return data_tmp, data_tmp_square

    for i in range(0, tmp_len // 16):
        data_tmp1 = data1[..., 16 * i: 16 * (i + 1)]
        data_tmp, data_tmp_square = sum_fp16_lib(
            data_tmp1, data_tmp, data_tmp_square)

    if tmp_len % 16 != 0:
        if tec_num > 1:
            data_tmp1 = np.zeros(
                [data1.shape[0], tec_num, 16], dtype=np.float32)
        else:
            data_tmp1 = np.zeros([data1.shape[0], 16], dtype=np.float32)
        tail_num = tmp_len % 16
        data_tmp1[..., 0: tail_num] = data1[..., tmp_len // 16 * 16:]
        data_tmp, data_tmp_square = sum_fp16_lib(
            data_tmp1, data_tmp, data_tmp_square)

    if len(data_tmp_square) > 0:
        return data_tmp, data_tmp_square
    return data_tmp


# align to lib, calc 8 or 16 data at a time
def sum_fp16(data1, tmp_len, data_tmp_sum, data_tmp_square=[], calc_num=16):
    for i in range(0, tmp_len // calc_num):
        data_tmp1 = data1[..., calc_num * i: calc_num * (i + 1)]
        data_tmp_sum += data_tmp1
        if len(data_tmp_square) > 0:
            data_tmp_square += data_tmp1 ** 2

    if tmp_len % calc_num != 0:
        data_tmp1 = np.zeros([data1.shape[0], calc_num], dtype=np.float32)
        tail_num = tmp_len % calc_num
        data_tmp1[..., 0: tail_num] = data1[...,
                                            tmp_len // calc_num * calc_num:]
        data_tmp_sum += data_tmp1
        if len(data_tmp_square) > 0:
            data_tmp_square += data_tmp1 ** 2
    if len(data_tmp_square) > 0:
        return data_tmp_sum, data_tmp_square
    return data_tmp_sum


# calc exp fp, use Taylor's Formula align to compiler
def fexp_tpc(x):
    def mad(a, b, c):
        import ctypes
        import platform
        libm = ctypes.CDLL("libm.so.6" if platform.system()
                           == "Linux" else None)
        if libm:
            fmaf_func = libm.fmaf
            fmaf_func.argtypes = [ctypes.c_float,
                                  ctypes.c_float, ctypes.c_float]
            fmaf_func.restype = ctypes.c_float
        fma = np.frompyfunc(lambda x, y, z: fmaf_func(x, y, z), 3, 1)
        return fma(a, b.astype('float32'), c)

    def convert_int8(x):
        # rtz
        # y = np.round(x)
        # flag = np.abs(x - x.astype('int32')) == 0.5
        # y[flag] = x.astype('int32')[flag]
        y = x.astype('int32')  # trunc in default
        return y

    def convert_float8(x):
        return x.astype('float32')

    def native_divide(x, y):
        return x / y

    def hex2fp32(x_str):
        return np.asarray(float.fromhex(x_str)).astype('float32')

    is_tensor = isinstance(x, torch.Tensor)
    x = x.float().cpu().numpy() if is_tensor else x.astype('float32')
    is_scalar = x.ndim == 0
    x = np.asarray([x]) if is_scalar else x

    ln2HI = hex2fp32('0x1.62e300p-1')
    ln2LO = hex2fp32('0x1.2fefa2p-17')
    invln2 = hex2fp32('0x1.715476p+0')
    zero = 0.0
    fhalF = np.ones_like(x) * 0.5
    fhalF[x < zero] = -0.5
    p = convert_int8(mad(x, invln2, fhalF))
    fp = convert_float8(p)
    hi = mad(fp, -ln2HI, x)  # t * ln2HI is exact here

    # Evaluate poly
    t = hi - fp * ln2LO
    tt = t * t

    tt0 = mad(tt, hex2fp32('0x1.637698p-25'), hex2fp32('-0x1.bbd41cp-20'))
    tt1 = mad(tt, tt0, hex2fp32('0x1.1566aap-14'))
    tt2 = mad(tt, tt1, hex2fp32('-0x1.6c16c2p-9'))
    tt3 = mad(tt, tt2, hex2fp32('0x1.555556p-3'))
    v = mad(tt, -tt3, t)

    y = 1.0 + t + native_divide(t * v, 2.0 - v)

    # Scale by 2^p, r = y * (2 ** p)
    r = (y * np.power(2.0, p.astype('float64'))).astype('float32')

    # ln(largest_normal) = 88.7228390520683530536
    ulim = hex2fp32('0x1.62e430p+6')
    # ln(smallest_normal) = -87.33654475055310898657
    llim = hex2fp32('-0x1.5d589ep+6')

    r[x < llim] = zero
    r[x >= ulim] = np.inf
    r[np.isnan(x)] = x[np.isnan(x)]
    r = torch.from_numpy(r) if is_tensor else r
    r = r[0] if is_scalar else r
    return r
