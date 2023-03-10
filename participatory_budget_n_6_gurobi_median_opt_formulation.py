import cvxpy as cp
import numpy as np
import math,random

import gurobipy as gp
from gurobipy import GRB

import itertools
from itertools import combinations, chain
 
def findsubsets(s, n):
    return list((map(frozenset, itertools.combinations(s, n))))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


n = 6




mapping_3_subset_list = findsubsets(range(n),3)
mapping_all_subset_list = list(powerset(range(n)))
mapping_3_subset_dict = {}###use this for mapping set to a corresponding index 
mapping_all_subset_dict = {}###use this for mapping set to a corresponding index 

thresh = 1.80 ##finding the optimal val



count = 0
for i in mapping_3_subset_list:
    mapping_3_subset_dict[i]= count
    count += 1

count = 0
for i in mapping_all_subset_list:
    mapping_all_subset_dict[frozenset(i)]=count
    count += 1

    


current_max = 0
current_idx = 0

###Note that $P$ denotes $[6]$ here.

###random_list = np.random.choice(2**(ncr(n,3)), 1000)

##print ("random list",random_list)

non_perm_list = [0, 1, 3, 7, 15, 19, 20, 21, 23, 28, 29, 31, 262135, 54, 55, 58, 59, 62, 63, 89565, 89567, 3272, 3274, 3279, 126, 127, 89580, 16311, 89581, 89582, 89583, 262140, 16315, 16318, 262141, 183, 184, 185, 16319, 187, 191, 207, 220, 221, 223, 262143, 89599, 8190, 254, 255, 11448, 11464, 79853, 79855, 11470, 11475, 11480, 11482, 11484, 97783, 524286, 16375, 97787, 97788, 16377, 97789, 524287, 97791, 495, 16382, 16383, 511, 228061, 228079, 228094, 228095, 120671, 3439, 32751, 120678, 120679, 96751, 32759, 120686, 120687, 1048575, 32764, 32765, 32767, 1023, 120694, 120695, 1043, 1044, 1045, 1047, 1052, 1053, 1055, 1062, 1063, 1064, 1065, 1066, 1067, 1070, 1071, 1078, 1079, 11612, 1082, 1083, 1084, 1085, 1086, 1087, 11619, 11620, 11621, 11622, 1132, 1133, 1134, 1135, 11624, 11625, 1150, 1151, 11628, 1188, 1189, 1191, 1192, 1193, 1195, 1196, 1197, 1199, 1207, 1208, 1209, 11638, 1211, 1212, 1213, 1215, 11639, 1228, 1229, 1231, 1244, 1245, 1247, 1260, 1261, 1262, 1263, 1278, 1279, 1516, 1517, 1519, 1535, 1536, 1537, 1539, 1541, 1543, 81719, 1551, 1555, 1556, 1557, 1559, 1564, 1565, 1567, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1590, 1591, 1594, 1595, 1596, 1597, 1598, 1599, 120812, 120813, 120815, 1644, 1645, 1646, 1647, 1662, 1663, 1700, 1701, 1703, 1704, 1705, 1707, 1708, 1709, 1711, 1719, 1720, 1721, 98035, 1723, 1724, 1725, 1727, 98038, 98039, 1740, 1741, 1743, 1756, 1757, 1759, 98043, 1772, 1773, 1774, 1775, 98046, 1790, 1791, 11759, 81791, 228335, 65520, 65521, 65523, 65527, 81813, 2028, 2029, 2031, 81815, 65535, 2047, 81823, 98127, 81846, 81847, 98131, 81849, 98132, 81850, 98133, 81851, 98135, 81854, 98137, 81855, 98139, 98140, 98141, 98143, 81867, 81870, 81871, 76991, 98159, 81883, 98166, 98167, 81885, 81886, 81887, 98170, 98171, 98173, 98174, 98175, 90038, 90039, 90042, 90043, 90044, 90045, 90046, 90047, 77023, 90055, 81918, 81919, 90061, 90062, 90063, 90070, 90071, 241500, 241501, 90077, 90079, 90095, 229375, 241534, 90110, 241535, 90111, 98279, 261103, 98281, 98283, 98285, 98287, 261111, 98291, 98292, 98293, 77128, 98295, 98296, 98297, 98299, 98300, 98301, 98303, 243695, 3126, 3127, 3130, 3131, 3134, 3135, 77151, 122742, 122743, 3164, 3165, 3166, 3167, 122750, 122751, 3198, 3199, 3219, 3220, 3221, 3223, 3224, 3225, 77167, 3227, 3228, 3229, 3231, 11443, 3254, 11447, 76983, 76984, 11449, 3259, 76985, 11451, 76987, 3263, 11455, 3257, 3258, 11458, 11459, 3262, 11462, 11463, 77000, 11465, 77001, 11466, 77002, 11467, 77003, 77006, 11471, 77007, 3273, 3275, 3278, 11477, 11478, 11479, 11476, 11481, 77018, 11483, 77019, 77020, 11485, 77021, 11486, 77022, 11487, 3290, 3291, 3292, 3293, 3294, 3295, 11507, 11510, 11511, 11512, 11513, 11514, 11515, 3326, 3327, 11518, 11519, 77054, 77055, 3347, 3348, 3349, 3350, 3351, 3356, 3357, 3359, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3400, 77129, 3401, 77131, 77132, 77133, 3402, 77135, 3403, 3404, 3405, 3406, 11600, 11601, 3407, 11603, 11604, 11605, 11607, 77147, 77148, 11613, 77149, 3423, 11615, 3418, 3419, 3420, 3421, 3422, 11616, 11623, 11617, 11618, 11626, 11627, 77164, 11629, 77165, 11630, 77166, 11631, 3436, 3437, 3438, 11632, 11633, 11634, 11636, 11640, 11635, 11643, 11637, 11645, 11646, 11641, 3454, 11642, 11644, 3455, 11647, 77182, 77183, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3546, 3547, 3548, 3549, 3550, 3551, 11744, 11745, 11747, 11748, 11749, 11751, 11752, 11753, 11755, 3564, 3565, 3566, 3567, 11760, 11761, 11756, 11763, 11764, 11765, 11757, 11767, 11768, 11769, 11771, 11772, 11773, 3582, 3583, 11775, 77311, 122835, 122836, 122837, 122839, 122841, 122843, 122844, 122845, 122847, 122866, 122867, 122870, 122871, 122873, 122874, 122875, 122878, 122879, 3840, 3841, 3842, 3843, 3846, 3847, 3849, 3851, 77292, 3855, 77293, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 77295, 3894, 3895, 3896, 3897, 3898, 3899, 3902, 3903, 3912, 77641, 3913, 77643, 3914, 3915, 77640, 3918, 3919, 12112, 12113, 12115, 12116, 12117, 77647, 12119, 12120, 12121, 3930, 12123, 77660, 12125, 3931, 3932, 3933, 3934, 12124, 3935, 12127, 77661, 77663, 241503, 241514, 241515, 241518, 241519, 12144, 12145, 12146, 12147, 12150, 12151, 12152, 12153, 12154, 12155, 3966, 3967, 3968, 3969, 3970, 3971, 12158, 12159, 3974, 3975, 3976, 3977, 3978, 3979, 3982, 3983, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 131056, 131057, 4022, 4023, 4024, 4025, 4026, 4027, 4030, 4031, 131059, 4040, 4041, 4042, 4043, 4046, 4047, 131063, 131064, 4058, 4059, 4060, 4061, 4062, 4063, 131065, 520157, 520159, 131067, 241643, 241647, 12272, 12273, 12275, 12279, 12280, 12281, 12283, 131071, 4094, 4095, 12287, 77823, 241663, 520191, 12486, 12487, 78030, 12495, 78031, 12502, 12503, 12508, 12509, 78044, 12511, 78046, 78045, 78047, 12534, 12535, 12536, 12537, 12538, 12539, 12542, 12543, 78078, 78079, 12576, 12577, 12579, 12581, 78117, 12583, 78119, 78121, 78123, 78125, 12591, 78127, 12593, 12595, 12596, 12597, 12598, 12599, 78134, 78137, 78135, 78139, 12604, 12605, 78140, 12607, 78141, 78142, 78143, 78157, 78158, 78159, 78172, 78173, 78174, 78175, 12640, 12641, 12642, 12643, 12645, 12646, 12647, 12649, 12651, 78189, 78190, 78191, 12655, 12657, 12658, 12659, 12660, 12661, 12662, 12663, 12664, 12665, 12666, 12667, 12668, 12669, 12670, 12671, 78206, 78207, 78245, 78246, 78247, 78249, 78251, 78253, 78254, 78255, 78262, 78263, 78265, 78267, 78268, 78269, 78270, 78271, 245758, 78285, 78286, 78287, 243511, 78300, 78301, 78302, 78303, 12768, 12769, 12771, 12773, 12774, 12775, 228863, 78317, 78318, 12783, 12784, 12785, 78319, 12787, 12788, 12789, 12790, 12791, 96979, 12796, 12797, 78334, 12799, 78335, 96982, 96983, 96987, 97010, 97011, 97013, 97014, 97015, 97017, 97018, 243558, 97019, 243559, 97021, 243566, 243567, 78593, 78595, 77694, 78599, 78601, 78603, 78607, 78612, 78613, 78614, 78615, 78617, 78619, 78620, 78621, 78622, 78623, 77695, 242467, 243579, 242471, 242479, 13104, 13105, 13107, 13110, 13111, 78648, 78649, 78650, 78646, 78647, 78651, 78654, 13119, 78655, 242495, 119623, 78665, 78667, 78670, 78671, 119631, 119639, 14010, 78683, 78684, 78685, 78686, 78687, 119647, 242543, 13168, 13169, 13170, 13171, 14015, 13174, 13175, 119671, 13177, 13179, 78718, 13183, 78719, 119679, 242559, 229118, 78727, 78729, 78731, 78734, 78735, 78740, 78741, 78742, 78743, 78745, 78747, 78748, 78749, 78750, 78751, 229119, 78774, 78775, 78776, 78777, 78778, 78779, 78782, 78783, 78793, 78795, 78798, 78799, 78811, 78812, 78813, 78814, 78815, 13296, 13297, 13299, 13302, 13303, 119799, 78846, 13311, 78847, 119807, 242687, 79663, 13526, 13527, 79215, 79068, 13533, 79069, 79070, 13535, 13532, 79071, 13542, 13543, 13544, 13545, 13546, 13547, 13550, 13551, 79086, 79087, 13558, 13559, 13560, 13561, 13562, 13563, 13564, 13565, 13566, 13567, 79102, 79103, 245742, 243683, 13619, 13620, 13621, 13623, 79159, 79163, 13628, 79165, 13629, 79164, 13631, 79167, 243687, 79196, 79197, 79198, 79199, 243693, 13666, 13667, 13668, 13669, 13670, 13671, 13672, 13673, 13674, 13675, 79212, 13677, 13678, 13679, 79213, 79214, 13676, 13683, 13684, 13685, 13686, 13687, 13688, 13689, 13690, 13691, 13692, 13693, 13694, 13695, 79230, 79231, 243699, 243702, 243703, 79269, 79271, 79276, 79277, 79278, 79279, 79287, 79292, 79293, 79294, 79295, 79308, 79309, 79310, 79311, 79324, 79325, 79326, 79327, 13792, 13793, 13795, 13796, 13797, 13798, 13799, 13804, 13805, 79340, 13807, 13808, 13809, 79342, 13811, 13812, 13813, 13814, 13815, 79343, 13820, 13821, 79358, 13823, 79359, 77659, 13906, 13907, 13910, 13911, 13914, 13915, 79451, 13918, 13919, 79454, 79455, 13938, 13939, 13942, 13943, 13944, 13945, 13946, 13947, 13948, 13949, 13950, 13951, 79486, 15799, 79487, 79511, 79513, 229101, 79515, 79517, 229102, 79519, 229103, 14001, 14002, 14003, 14004, 14005, 79542, 14007, 79544, 14009, 79545, 79546, 14011, 79548, 14013, 14006, 14008, 14012, 14016, 14017, 14019, 14014, 14021, 14022, 14023, 79561, 79563, 79564, 79565, 79566, 14031, 14032, 14033, 79567, 14035, 14036, 14037, 14038, 14039, 245759, 79579, 79580, 14045, 14044, 14047, 79582, 14049, 14050, 14051, 14052, 14053, 14054, 14055, 14056, 14057, 14058, 14059, 79596, 14060, 14062, 14061, 14063, 14065, 14066, 14067, 14068, 14069, 14070, 14071, 14072, 14073, 14074, 14075, 14076, 14077, 14078, 14079, 79614, 79615, 79637, 79639, 79641, 79643, 79644, 79645, 79646, 79647, 14112, 14113, 14115, 14117, 79654, 79655, 14119, 79656, 79653, 79659, 79657, 79661, 79662, 79660, 79658, 14127, 14128, 14129, 14132, 14131, 79670, 14135, 14133, 79673, 79674, 14134, 14140, 79672, 14141, 79676, 14143, 79675, 79677, 79678, 79679, 243519, 79689, 79691, 79692, 79693, 79694, 79695, 79341, 120663, 79707, 79708, 79709, 79710, 79711, 14176, 14177, 14178, 14179, 14180, 14181, 14182, 14183, 14184, 14185, 14186, 14187, 79724, 14188, 14189, 14190, 14192, 14193, 14194, 14195, 14196, 14197, 14198, 14191, 14200, 14201, 14202, 14203, 14204, 14205, 14199, 14206, 14207, 79742, 79743, 120702, 120703, 243582, 243583, 79753, 79755, 97263, 79757, 79759, 79765, 79767, 15855, 79772, 79773, 79774, 79775, 79781, 79782, 79783, 79784, 79785, 79786, 79787, 79788, 79789, 79790, 79791, 79798, 79799, 79800, 79801, 79802, 79803, 79804, 79805, 79806, 79807, 79817, 79819, 79820, 79821, 79822, 79823, 79835, 79836, 79837, 79838, 79839, 14304, 14305, 14307, 14308, 14309, 14310, 14311, 120805, 120803, 120807, 120811, 14316, 14317, 79852, 14319, 14320, 14321, 79854, 14323, 14324, 14325, 14326, 14327, 120820, 120819, 120821, 120823, 14332, 14333, 79870, 14335, 120828, 79871, 120827, 120829, 243708, 120831, 243709, 243711, 32243, 32245, 32247, 88517, 88519, 88525, 88527, 88534, 88535, 88540, 88541, 88542, 88543, 31200, 31201, 31203, 31205, 96741, 31207, 96743, 96745, 96747, 88557, 88558, 31215, 88559, 31217, 96749, 31219, 31220, 31221, 96757, 31223, 96759, 96761, 96763, 31228, 31229, 88574, 96764, 31231, 88575, 96765, 96767, 227839, 88662, 88663, 88670, 88671, 88702, 88703, 228023, 228025, 228027, 228029, 228031, 88770, 88771, 88775, 88778, 88779, 88782, 88783, 31440, 31441, 228047, 31443, 88788, 88789, 31446, 88790, 31447, 88793, 88794, 88795, 88796, 88797, 88791, 88798, 96986, 96985, 31455, 88799, 96990, 96991, 88806, 88807, 228063, 88810, 88811, 88814, 88815, 31472, 31473, 31474, 31475, 88820, 88821, 31477, 88822, 31478, 88825, 88826, 31482, 88828, 97016, 88823, 88829, 31479, 97020, 88827, 31483, 88830, 31487, 88831, 97022, 97023, 97091, 97095, 79543, 97099, 97103, 97107, 97108, 97109, 97111, 97113, 97115, 97116, 79547, 97117, 97119, 97122, 97123, 97126, 79549, 97127, 97130, 97131, 79550, 97133, 97134, 97135, 79551, 245630, 97139, 97140, 97141, 97142, 97143, 245631, 97145, 97146, 97147, 97148, 97149, 97150, 97151, 88979, 88981, 88983, 88985, 88986, 88987, 88988, 88989, 88990, 88991, 229357, 88999, 89001, 89003, 229359, 89005, 89007, 89012, 89013, 89014, 89015, 89018, 89019, 89020, 89021, 89022, 89023, 89029, 89030, 89031, 89037, 89038, 89039, 89046, 89047, 89052, 89053, 89054, 89055, 97249, 97251, 261091, 97253, 97255, 261095, 97257, 97259, 89069, 89070, 89071, 31728, 31729, 97261, 31731, 97268, 31733, 97269, 31735, 97272, 97267, 97273, 97271, 97276, 97277, 89086, 31743, 89087, 97275, 97279, 228351, 261117, 261119, 79581, 79583, 79597, 79598, 79599, 7294, 7295, 245687, 245691, 245693, 7350, 7351, 7352, 7353, 7354, 7355, 245695, 7358, 7359, 81375, 7368, 7369, 7370, 7371, 7374, 7375, 7386, 7387, 7388, 7389, 7390, 7391, 15606, 15607, 15608, 15609, 15610, 15611, 7422, 7423, 15614, 15615, 81150, 81151, 245724, 245725, 245727, 81260, 81261, 81262, 81263, 15732, 15733, 15734, 15735, 227823, 15740, 15741, 15742, 15743, 81278, 81279, 245738, 7571, 7572, 7573, 7575, 245739, 7580, 7581, 7583, 7588, 7589, 7590, 7591, 7592, 7593, 7594, 7595, 7596, 7597, 7598, 7599, 242487, 245743, 15795, 15796, 15797, 7606, 7607, 15800, 81336, 7610, 7611, 7612, 7613, 7614, 15801, 15804, 7615, 81337, 15803, 15805, 81340, 15807, 81343, 81353, 81355, 81356, 81357, 81358, 79671, 81359, 15827, 15828, 15829, 15830, 15831, 89559, 81371, 15836, 15837, 81372, 89564, 15839, 81374, 15842, 15843, 15844, 15845, 15846, 15847, 15848, 15849, 15850, 15851, 15852, 7660, 7661, 7662, 7663, 15853, 81388, 15859, 15860, 15861, 15862, 15854, 15864, 15865, 15866, 32244, 15868, 15863, 7678, 7679, 15867, 15869, 15870, 32252, 15871, 32253, 81406, 89598, 32255, 81390, 81391, 89726, 89727, 81335, 81339, 81341, 8127, 229047, 229048, 229049, 229051, 229052, 229053, 229055, 32503, 229069, 229071, 89812, 89813, 89814, 89815, 89818, 89819, 89820, 79725, 89821, 89822, 89823, 229084, 79726, 229085, 229087, 89829, 89830, 79727, 89831, 89834, 89835, 89836, 89837, 89838, 89839, 32496, 32497, 32498, 32499, 89844, 89845, 89846, 32502, 32504, 32505, 32506, 98040, 89852, 89850, 89847, 89853, 89851, 32507, 98041, 98042, 32510, 89854, 32511, 89855, 98047, 81683, 81684, 81685, 81686, 81687, 81688, 81689, 81690, 81691, 81692, 81693, 81694, 81695, 16176, 16177, 16178, 16179, 16182, 16183, 81720, 16185, 16186, 16184, 81721, 16187, 16190, 81718, 81722, 16191, 81723, 98115, 81373, 98117, 81726, 98119, 81727, 81737, 81738, 81739, 98123, 98125, 81742, 81743, 16208, 16209, 16210, 16211, 16212, 16213, 16214, 16215, 16216, 16217, 16218, 16219, 16220, 16221, 16222, 81754, 81756, 16223, 81755, 98147, 81757, 98149, 98150, 81758, 81759, 98153, 98154, 98151, 98156, 98155, 98157, 98158, 16240, 16241, 16242, 16243, 98164, 98163, 16246, 16247, 16248, 16249, 16250, 16251, 98165, 98169, 16254, 16255, 8064, 8065, 8066, 8067, 81790, 98172, 8070, 8071, 81407, 8078, 8079, 81389, 8082, 8083, 8084, 8085, 8086, 8087, 81812, 81816, 81814, 81811, 8092, 8093, 8094, 8095, 81818, 81817, 81820, 81822, 81819, 90021, 81821, 90023, 90024, 90025, 90026, 90027, 90029, 90031, 16304, 16305, 16306, 16307, 90036, 90037, 8118, 8119, 16312, 16310, 8122, 8123, 16314, 16313, 8126, 81848, 16320, 16321, 16322, 16323, 90052, 90053, 16326, 16327, 81864, 16329, 81866, 16331, 90060, 81865, 90054, 16335, 16336, 16337, 16338, 16339, 16340, 16341, 16342, 16343, 16344, 16345, 16346, 16347, 16348, 81882, 16350, 16349, 32736, 32737, 90076, 32739, 81884, 32741, 16351, 90078, 98280, 32743, 98275, 98277, 90092, 90093, 90094, 98284, 16368, 16369, 16370, 32752, 16371, 32753, 16374, 32756, 16376, 3255, 16378, 16379, 32755, 32757, 3256, 8191]
##denotes the list of cases removed after permuting voter preferences obtained from output of file permutations.py




for index in range(len(non_perm_list))[::-1]:

    itr = non_perm_list[index]
    model = gp.Model('RAP')
    model.setParam('NonConvex', 2)
    model.setParam('MIPGapAbs',5e-7)
    ##model.setParam('MIPFocus', 1)
    ##model.setParam('MIPGap',1.2)
   
###First part common for all no use of itr
    X= model.addVars(2**n, vtype=GRB.CONTINUOUS,name= "X_name") ##denotes X(S) for every subset of $[n]$. 


    
    Z = model.addVars(ncr(n,3),2**n, vtype=GRB.CONTINUOUS,name = "Z_name")

    V = model.addVars(2**n, vtype=GRB.CONTINUOUS,name = "V_name") ##denotes V(S) for every subset of $[n]$. 

    
    
    


    for i in range(n):
        temp_S = set(range(n))- set({i}) ### set of all elements excluding voter $i$ 
        
        
        temp_list= list(powerset(temp_S)) ###powerset of all elements excluding voter $i$
        
        testing_list = [] ###stores X(iUS) for all subsets of S in [n]-i

        for j in temp_list:
          testing_list.append(X[mapping_all_subset_dict[frozenset(set(j)|set({i}))]])
        
        
        
        model.addConstr(gp.quicksum(testing_list) == 1,name="individualvote"+str(i)) ###\sum_{S \ni i} X(S) = 1

       

    model.addConstrs((Z.sum(j,'*') == 1 for j in range(ncr(n,3)))) ## for every subset of budgets, \{\{x,y\},c}\, we have $\sum_{S \in \mathcal{P}(P)} Z^{\c,\{x,y\}}(S) = 1$
    model.addConstr(V.sum('*') == 1) ### \sum_{S \in \mathcal{P}([6])} V[S] = 1
    model.addConstrs((X[j]>=V[j] for j in range(2**n))) ## X(S) \geq V(S)
    
    for i in range(ncr(n,3)):
        R_set = set(mapping_3_subset_list[i]) ## denotes the precise set $Q$ chosen
        R_set_list = list(R_set)
        
        Q_set_excl = list(powerset(set(range(n))-R_set)) ###power set of the set $[n]-Q$
        
       
        model.addConstrs((X[j]>=Z[i,j] for j in range(2**n))) ##denoting X(S) \geq Z^{\{x,y\},c} 
            
        
        
        
        for j in Q_set_excl: ###considers every set $Q$
            
                model.addConstr(Z[i,mapping_all_subset_dict[frozenset(set(j)|R_set)]]== X[mapping_all_subset_dict[frozenset(set(j)|R_set)]]) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing all elts of $Q$.
            
            
                model.addConstr(Z[i,mapping_all_subset_dict[frozenset(set(j))]]==0) ##Z^{\x,y\,c}(S) = X(S) for every $S$ containing no elements of $Q$.
            
            
               
            
            
            
              

                    
            
            
            
            
            
    list_temp1 = [] ###calculation the distance of bargaing solution Z with other voters not ones in Q excluding constant 1


    for R_index in range(ncr(n,3)):
        for i in set(range(n))-set(mapping_3_subset_list[R_index]):
            for S in powerset(set(range(n))-set({i})):
                
                    list_temp1.append(-2*Z[R_index,mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
    list_temp_const_1 = []
    list_temp_const_2 = []

    for R_index in range(ncr(n,3)): ##calculating the constants 
        for i in set(range(n))-set(mapping_3_subset_list[R_index]):
            
                list_temp_const_1.append(2)


                

    list_temp2 = [] ###calculation the distance of optimal solution with all voters excluding constant 1
    for i in range(n): ##distance with voter $i$
        for S in powerset(set(range(n))-set({i})): 
            list_temp2.append(-2*V[mapping_all_subset_dict[frozenset(set(S)|set({i}))]])
            
    for i in range(n): ##constants
        list_temp_const_2.append(2)
            
    
    model.setObjective((1.0/(ncr(n,3)*(n-3)))*(gp.quicksum(list_temp1)+gp.quicksum(list_temp_const_1)) - thresh*(1.0/n)*(gp.quicksum(list_temp2)+gp.quicksum(list_temp_const_2)),GRB.MAXIMIZE)




    ####Second part which uses itr begins here












    
    list_diff_R = format(itr, '0'+str(ncr(n,3))+'b') ###analysing every case in $\mathbb{K}$ after removing the directly toggled cases and those unique to permutation.
    
    ##representation in binary format

    for j in range(ncr(n,3)): ###denoting which set R is considered
        list_temp3 = [] ### computing X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c. 
        for i in range(2,4):
                for R_prime in findsubsets(set(mapping_3_subset_list[j]),i):
                    for V_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                        list_temp3.append(X[mapping_all_subset_dict[frozenset(set(R_prime)|set(V_prime))]])
        
        if (list_diff_R[j]=='1'): ## considers Case 2 as defined in every condition. 
            model.addConstr(gp.quicksum(list_temp3) >=1) ##puts desired constraint on X(xy) + X(xyc) + X(xc)+ X(yc) on a incremental space of x,y,c 
            for R_prime in findsubsets(set(mapping_3_subset_list[j]),1): ### identifying the disagreement point
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      
                        model.addConstr(Z[j,mapping_all_subset_dict[frozenset(set(R_prime)|set(U_prime))]]==0) ## Z(x)=Z(y)= Z(c) = 0 in the incremental budget space

                        
                        
            ##print("entered and finished part1",j)
        elif (list_diff_R[j]=='0'):
            ##print ("entering here",list_diff_R[j])
            model.addConstr(gp.quicksum(list_temp3) <=1)
            for R_prime in findsubsets(set(mapping_3_subset_list[j]),2):
                    for U_prime in powerset(set(range(n))-set(mapping_3_subset_list[j])):
                      
                        model.addConstr(Z[j,mapping_all_subset_dict[frozenset(set(R_prime)|set(U_prime))]]==X[mapping_all_subset_dict[frozenset(set(R_prime)|set(U_prime))]]) ## Z(xy) = X(xy); Z(yc)= X(yc); Z(xc)=X(xc) = 0 in the incremental budget space


                        
            ##print("entered and finished part2",j)
        else:
            print ("The list contains spaces neither 0 nor 1")
            break
            break
 
    
    model.write("RAP.lp")
    print ("itr")
    model.optimize()
    
    if (model.status != 2):
            print (model.status,"iteration",itr,"not solved")

            file1 = open("myfile_gurobi_median_upto_toggle_updated.txt","a")
            file1.writelines(str(model.status)+ " case "+str(itr)+ " itr " + str(index)+ " not solved \n",)
            file1.close()
            
            ##print (model.status,"iteration",itr,)
    
    if (model.status == 2):
        file1 = open("myfile_gurobi_median_upto_toggle_updated.txt","a")
        file1.writelines(str(model.status)+ " case "+str(itr)+ " itr " + str(index)+ " solved " +" Obj Value " + str (model.objVal)+"\n")
        file1.close()


    if (itr % 1 == 0):
        
            print ("Current max", current_max,"max_idx",current_idx, "iteration",index,"case number",itr)
            
    if (model.objVal > current_max):
      current_max = max(model.objVal,current_max) ##checking maximum value obtained
      current_idx = itr

      

    
    if (current_max > 5e-6):
      print ("Not working", current_max)
    
print ("Final maximum obtained",current_max,"max_idx",current_idx)

          


###random_list = [213]








##prob.solve()


