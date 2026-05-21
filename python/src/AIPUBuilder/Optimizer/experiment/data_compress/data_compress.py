# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
# input compressed 256 bit


def decompress_8bits(compressdata, lut, basebit, herit_last_frame, remain_no_decompress_cnt):
    prefix = ''
    head = '1'*basebit
    prehead = head
    decodeData = []
    level = 0
    bits_each_read = basebit
    class_num = (1 << basebit) - 1
    header_len = 5
    tai_len = int(compressdata[0:header_len][::-1], 2)
    new_compresssdata = herit_last_frame + compressdata[header_len:]
    processed_bits_in256 = 0
    cnt = 0
    one_time_bit_cnt = 4
    while ((processed_bits_in256 != len(herit_last_frame) + 251-tai_len) and (cnt < remain_no_decompress_cnt)):
        prefix = prefix + \
            new_compresssdata[processed_bits_in256 +
                              one_time_bit_cnt:processed_bits_in256+one_time_bit_cnt+bits_each_read]
        if prefix == prehead and len(prefix) % basebit == 0:
            level = level + 1
            one_time_bit_cnt = one_time_bit_cnt + bits_each_read
            prehead = prehead + '1'*bits_each_read
        else:
            offset = (class_num*level)
            index = offset + \
                int(new_compresssdata[processed_bits_in256+one_time_bit_cnt:
                    processed_bits_in256 + bits_each_read+one_time_bit_cnt][::-1], 2)
            one_time_bit_cnt = one_time_bit_cnt + bits_each_read
            # merge higher4bit and lower4bit
            lsb4bit = int(new_compresssdata[processed_bits_in256:processed_bits_in256+4][::-1], 2)
            bdata = lut[0, index]*16 + lsb4bit
            decodeData.append(bdata)
            level = 0
            prehead = head
            prefix = ''
            cnt = cnt + 1
            processed_bits_in256 = processed_bits_in256 + one_time_bit_cnt
            one_time_bit_cnt = 4
    return np.array(decodeData), compressdata[256-tai_len:256]

# input compressed 256 bit


def decompress_4bits(compressdata, lut, basebit, herit_last_frame, remain_no_decompress_cnt):
    prefix = ''
    head = '1'*basebit
    prehead = head
    decodeData = []
    level = 0
    bits_each_read = basebit
    class_num = (1 << basebit) - 1
    header_len = 5
    tai_len = int(compressdata[0:header_len][::-1], 2)
    new_compresssdata = herit_last_frame + compressdata[header_len:]
    processed_bits_in256 = 0
    cnt = 0
    while ((processed_bits_in256 != len(herit_last_frame) + 251-tai_len) and (cnt < remain_no_decompress_cnt)):
        prefix = prefix + \
            new_compresssdata[processed_bits_in256:processed_bits_in256+bits_each_read]
        if prefix == prehead and len(prefix) % basebit == 0:
            level = level + 1
            processed_bits_in256 = processed_bits_in256 + bits_each_read
            prehead = prehead + '1'*bits_each_read
        else:
            offset = (class_num*level)
            index = offset + \
                int(new_compresssdata[processed_bits_in256:
                    processed_bits_in256 + bits_each_read][::-1], 2)
            processed_bits_in256 = processed_bits_in256 + bits_each_read
            decodeData.append(lut[0, index])
            level = 0
            prehead = head
            prefix = ''
            cnt = cnt + 1
    return np.array(decodeData), compressdata[256-tai_len:256]

# compress data ,get 256bit compressed bit stream


def data8_compress_256bits(from_last_step_bits, id, l4bit, width, index_str_list, lsb_str_list):
    compressed_data = from_last_step_bits
    class_num = (1 << width)-1
    level = id//class_num
    index_data = (id % class_num)*width
    base_one_str = '1'*width
    bit_cnt = len(compressed_data)
    header = '00000'
    i = 0
    while(bit_cnt < 251 and i < len(id)):
        # str_bin = np.binary_repr(index_data[i], width=width)
        str_bin = index_str_list[index_data[i]:index_data[i]+width]
        # compress high 4bits
        str_bin = base_one_str*level[i] + str_bin
        # copy lower 4bits, lower4bit then higher 4bit
        str_bin = lsb_str_list[l4bit[i]*4:l4bit[i]*4+4] + str_bin
        bit_cnt = bit_cnt + len(str_bin)
        compressed_data = compressed_data + str_bin
        i = i + 1
    # handle tail
    if i == len(id) and bit_cnt < 251:
        return '00000' + compressed_data, '', i

    # hw read 256bit each step and 5bits is header len.so valid bits = 251
    tail_len = 251 - len(compressed_data) + len(str_bin)
    header = np.binary_repr(tail_len, width=5)[::-1]
    compressed_data = header + compressed_data + str_bin
    to_next_step_bits = compressed_data[256:256+bit_cnt-251]

    return compressed_data[0:256], to_next_step_bits, i

# compress data ,get 256bit compressed bit stream


def data4_compress_256bits(from_last_step_bits, id, width, index_str_list):
    compressed_data = from_last_step_bits
    class_num = (1 << width)-1
    level = id//class_num
    index_data = (id % class_num)*width
    base_one_str = '1'*width
    bit_cnt = len(compressed_data)
    header = '00000'
    i = 0
    while((bit_cnt < 251) and (i < len(id)-1)):
        str_bin0 = index_str_list[index_data[i]:index_data[i]+width]
        # compress  4bits
        str_bin0 = base_one_str*level[i] + str_bin0
        bit_cnt = bit_cnt + len(str_bin0)
        compressed_data = compressed_data + str_bin0
        i = i + 1
        str_bin1 = index_str_list[index_data[i]:index_data[i]+width]
        # compress  4bits
        str_bin1 = base_one_str*level[i] + str_bin1
        bit_cnt = bit_cnt + len(str_bin1)
        compressed_data = compressed_data + str_bin1
        i = i + 1
    # handle tail
    if i == len(id) and bit_cnt < 251:
        return '00000' + compressed_data, '', i
    # hw read 256bit each step and 5bits is header len.so valid bits = 251
    tail_len = 251 - len(compressed_data) + len(str_bin0+str_bin1)
    header = np.binary_repr(tail_len, width=5)[::-1]
    compressed_data = header + compressed_data
    to_next_step_bits = compressed_data[256:256+bit_cnt-251]
    return compressed_data[0:256], to_next_step_bits, i

# input datac is 4bit data


def data4bit_compress(datac, verify_decode, spare_flag,):

    lut = np.zeros((1, 1 << 4), dtype=np.uint8)
    tot_len = len(datac)
    # simulate random 50% 0 sparse
    if spare_flag:
        mask0 = np.random.randint(0, 2, tot_len)
        datac = datac*mask0

    l4bit = (datac).astype(np.uint8)
    l4bit_stats = np.bincount(l4bit)

    l4bit_prob = l4bit_stats/l4bit_stats.sum()
    order4_l4bit = np.flip(np.argsort(l4bit_stats))
    l4bit_prob = l4bit_prob[order4_l4bit]
    lut[0, 0:order4_l4bit.size] = order4_l4bit

    l4cnt = l4bit_stats[order4_l4bit]
    bits_distrubte = - \
        np.round(np.log2(l4bit_prob+(l4bit_prob == 0))).astype(np.uint8)
    compressed_data = ''
    a = lut[0, :]
    # index of first occurrence of each value
    sorter = np.argsort(a)
    id = sorter[np.searchsorted(a, l4bit, sorter=sorter)]

    from_last_step_bits = ''
    consume_data = 0
    herit_last_frame = ''
    compr_need_bits = 0

    # need not compress
    compr_need_bits = tot_len*4
    need_compress = True
    if (bits_distrubte[0] <= 1):
        y1 = l4cnt[0:1].sum()
        y2 = l4cnt[1:2].sum()
        y3 = l4cnt[2:3].sum()
        y4 = l4cnt[3:4].sum()
        y5 = l4cnt[4:5].sum()
        y6 = l4cnt[5:6].sum()
        y7 = l4cnt[6:7].sum()
        y8 = l4cnt[7:8].sum()
        y9 = l4cnt[8:9].sum()
        y10 = l4cnt[9:10].sum()
        y11 = l4cnt[10:11].sum()
        y12 = l4cnt[11:12].sum()
        y13 = l4cnt[12:13].sum()
        y14 = l4cnt[13:14].sum()
        y15 = l4cnt[14:15].sum()
        y16 = l4cnt[15:16].sum()
        compr_need_bits = y1 + y2 * 2 + y3 * 3 + y4 * 4 + y5 * 5 + y6 * 6 + y7 * \
            7 + y8 * 8 + y9 * 9 + y10*10+y11*11+y12*12+y13*13+y14*14+y15*15+y16*16

        unit_shift = 1
        # # index is encoded to 0
        index_str_list = '0'

    elif (bits_distrubte[0] == 2):
        y1 = l4cnt[0:3].sum()
        y2 = l4cnt[3:6].sum()
        y3 = l4cnt[6:9].sum()
        y4 = l4cnt[9:12].sum()
        y5 = l4cnt[12:15].sum()
        y6 = l4cnt[15:16].sum()
        compr_need_bits = y1 * 2 + y2 * 4 + y3 * 6 + y4 * 8 + y5 * 10 + y6*12

        unit_shift = 2
        # index is encoded to 00,01,10
        # bit store format(lsb to msb )
        index_str_list = '001001'

    elif (bits_distrubte[0] == 3):
        unit_shift = 3
        # index is encoded to 000,001,10
        # bit store format(lsb to msb )
        index_str_list = '000100010110001101011'
        y1 = l4cnt[0:7].sum()
        y2 = l4cnt[7:14].sum()
        y3 = l4cnt[14:16].sum()
        compr_need_bits = y1 * 3 + y2 * 6 + y3 * 9
    else:
        compr_need_bits = tot_len*4

    if compr_need_bits < tot_len*4:
        while(consume_data < tot_len):
            compress_256bit, from_last_step_bits, encoded_cnt = data4_compress_256bits(
                from_last_step_bits, id[consume_data:consume_data+256], unit_shift, index_str_list)
            consume_data = consume_data + encoded_cnt
            compressed_data = compressed_data + compress_256bit
        if len(from_last_step_bits) > 0:
            compressed_data = compressed_data + '00000' + from_last_step_bits
        # verify decode if correct
        if verify_decode:
            decodedata = np.zeros(consume_data, dtype=np.uint8)
            frame_cnt = 0
            cnt = 0
            frame_cnt = len(compressed_data) >> 8
            for i in range(frame_cnt):
                decodedata_oneframe, herit_last_frame = decompress_4bits(
                    compressed_data[i*256:i*256+256], lut, unit_shift, herit_last_frame, tot_len-cnt)
                decodedata[cnt:cnt+len(decodedata_oneframe)] = decodedata_oneframe
                cnt = cnt + len(decodedata_oneframe)
            decodedata_oneframe, herit_last_frame = decompress_4bits(
                compressed_data[frame_cnt * 256:], lut, unit_shift, herit_last_frame, tot_len-cnt)
            decodedata[cnt:cnt + len(decodedata_oneframe)] = decodedata_oneframe
            cnt = cnt + len(decodedata_oneframe)
            if np.sum(decodedata[0:tot_len] == datac[0:tot_len]) != tot_len:
                print('decode dont match encode')
        compr_need_bits = len(compressed_data)
    # calcuate ideal compress bits according to info entrophy
    # t_cnt_bits = (
    #     ((-np.round(np.log2((l4bit_prob == 0)+l4bit_prob)))*l4bit_prob).sum())*tot_len

    return compressed_data, compr_need_bits


def data8bit_compress(datac, verify_decode, spare_flag):

    lut = np.zeros((1, 1 << 4), dtype=np.uint8)
    tot_len = len(datac)
    # simulate random 50% 0 sparse
    if spare_flag:
        mask0 = np.random.randint(0, 2, tot_len)
        datac = datac*mask0
    h4bit = (datac >> 4) & 0xf
    l4bit = datac & 0xf
    h4bit_stats = np.bincount(h4bit)

    h4bit_prob = h4bit_stats/h4bit_stats.sum()
    order4_h4bit = np.flip(np.argsort(h4bit_stats))
    h4bit_prob = h4bit_prob[order4_h4bit]
    lut[0, 0:order4_h4bit.size] = order4_h4bit

    h4cnt = h4bit_stats[order4_h4bit]
    bits_distrubte = - \
        np.round(np.log2(h4bit_prob+(h4bit_prob == 0))).astype(np.uint8)

    compressed_data = ''
    a = lut[0, :]
    # index of first occurrence of each value
    sorter = np.argsort(a)
    id = sorter[np.searchsorted(a, h4bit, sorter=sorter)]
    # bit store format(lsb to msb )
    lsb_str_list = '0000100001001100001010100110111000011001010111010011101101111111'
    from_last_step_bits = ''
    consume_data = 0
    herit_last_frame = ''
    compr_need_bits = 0
    if (bits_distrubte[0] <= 1):
        y1 = h4cnt[0:1].sum()
        y2 = h4cnt[1:2].sum()
        y3 = h4cnt[2:3].sum()
        y4 = h4cnt[3:4].sum()
        y5 = h4cnt[4:5].sum()
        y6 = h4cnt[5:6].sum()
        y7 = h4cnt[6:7].sum()
        y8 = h4cnt[7:8].sum()
        y9 = h4cnt[8:9].sum()
        y10 = h4cnt[9:10].sum()
        y11 = h4cnt[10:11].sum()
        y12 = h4cnt[11:12].sum()
        y13 = h4cnt[12:13].sum()
        y14 = h4cnt[13:14].sum()
        y15 = h4cnt[14:15].sum()
        y16 = h4cnt[15:16].sum()
        compr_need_bits = y1 + y2 * 2 + y3 * 3 + y4 * 4 + y5 * 5 + y6 * 6 + y7 * \
            7 + y8 * 8 + y9 * 9 + y10*10+y11*11+y12*12+y13*13+y14*14+y15*15+y16*16
        unit_shift = 1
        # index is encoded to 0
        index_str_list = '0'
    elif (bits_distrubte[0] == 2):
        y1 = h4cnt[0:3].sum()
        y2 = h4cnt[3:6].sum()
        y3 = h4cnt[6:9].sum()
        y4 = h4cnt[9:12].sum()
        y5 = h4cnt[12:15].sum()
        y6 = h4cnt[15:16].sum()
        compr_need_bits = y1 * 2 + y2 * 4 + y3 * 6 + y4 * 8 + y5 * 10 + y6*12
        unit_shift = 2
        # index is encoded to 00,01,10
        # bit store format(lsb to msb )
        index_str_list = '001001'
    else:
        compr_need_bits = tot_len*4
    if compr_need_bits < tot_len*4:
        while(consume_data < tot_len):
            compress_256bit, from_last_step_bits, encoded_cnt = data8_compress_256bits(
                from_last_step_bits, id[consume_data:consume_data+256], l4bit[consume_data:consume_data+256], unit_shift, index_str_list, lsb_str_list)
            consume_data = consume_data + encoded_cnt
            compressed_data = compressed_data + compress_256bit

        if len(from_last_step_bits) > 0:
            compressed_data = compressed_data + '00000' + from_last_step_bits

        # verify decode if correct
        if verify_decode:
            decodedata = np.zeros(consume_data, dtype=np.uint8)
            frame_cnt = 0
            cnt = 0
            frame_cnt = len(compressed_data) >> 8
            for i in range(frame_cnt):
                decodedata_oneframe, herit_last_frame = decompress_8bits(
                    compressed_data[i*256:i*256+256], lut, unit_shift, herit_last_frame, tot_len-cnt)
                decodedata[cnt:cnt+len(decodedata_oneframe)] = decodedata_oneframe
                cnt = cnt + len(decodedata_oneframe)

            decodedata_oneframe, herit_last_frame = decompress_8bits(
                compressed_data[frame_cnt * 256:], lut, unit_shift, herit_last_frame, tot_len-cnt)
            decodedata[cnt:cnt + len(decodedata_oneframe)] = decodedata_oneframe
            cnt = cnt + len(decodedata_oneframe)
            if np.sum(decodedata[0:tot_len] == datac[0:tot_len]) != tot_len:
                print('decode dont match encode')

        compr_need_bits = len(compressed_data)
    # calcuate ideal compress bits according to info entrophy
    # t_cnt_bits = (
    #     ((-np.round(np.log2((h4bit_prob == 0)+h4bit_prob)))*h4bit_prob).sum())*tot_len + tot_len*4

    return compressed_data, compr_need_bits
