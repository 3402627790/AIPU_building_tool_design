import numpy as np


### bitonic_sort releated function
def split_the_index(data):
    data1 = data[:16]
    data2 = data[16:]
    out1 = np.zeros([32])
    out2 = np.zeros([32])
    UXT2(out1, data1, 0)
    UXT2(out2, data2, 0)
    return out1, out2


def merge_the_index(data1, data2):
    out = np.zeros([32])
    for i in range(16):
        out[i] = data1[i * 2]
    for i in range(16):
        out[i + 16] = data2[i * 2]
    return out


def PERM(data, num):
    dataout = np.array(data)
    B2 = np.zeros([32])
    B3 = np.zeros([32])
    Tx = list(range(32))
    if num == 32:
        Tx = Tx[::-1]
        UXT(B2, B3, Tx)
        return np.array(data[::-1]), B2, B3
    elif num == 16:
        Tx = np.concatenate([Tx[16:], Tx[:16]])
        UXT(B2, B3, Tx)
        return np.concatenate([data[16:], data[:16]]), B2, B3
    elif num == 8:
        Tx = np.concatenate([Tx[8:16], Tx[:8], Tx[24:], Tx[16:24]])
        UXT(B2, B3, Tx)
        return np.concatenate([data[8:16], data[:8], data[24:], data[16:24]]), B2, B3
    elif num == 4:
        Tx = np.concatenate([Tx[4:8], Tx[0:4], Tx[12:16], Tx[8:12], Tx[20:24], Tx[16:20], Tx[28:], Tx[24:28]])
        UXT(B2, B3, Tx)
        return np.concatenate(
            [data[4:8], data[0:4], data[12:16], data[8:12], data[20:24], data[16:20], data[28:], data[24:28]]), B2, B3
    elif num == 2:
        for i in range(8):
            dataout[i * 4 + 2], dataout[i * 4] = dataout[i * 4], dataout[i * 4 + 2]
            dataout[i * 4 + 3], dataout[i * 4 + 1] = dataout[i * 4 + 1], dataout[i * 4 + 3]
            Tx[i * 4 + 2], Tx[i * 4] = Tx[i * 4], Tx[i * 4 + 2]
            Tx[i * 4 + 3], Tx[i * 4 + 1] = Tx[i * 4 + 1], Tx[i * 4 + 3]
        UXT(B2, B3, Tx)
        return dataout, B2, B3
    elif num == 1:
        for i in range(16):
            dataout[i * 2], dataout[i * 2 + 1] = dataout[i * 2 + 1], dataout[i * 2]
            Tx[i * 2], Tx[i * 2 + 1] = Tx[i * 2 + 1], Tx[i * 2]
        UXT(B2, B3, Tx)
        return dataout, B2, B3


def UXT(Td1, Td2, Tn):  # Sel control low half or high half
    for i in range(16):
        Td1[2 * i] = Tn[i]
        Td1[2 * i + 1] = 0  # int16 high 8bit
        Td2[2 * i] = Tn[i + 16]
        Td2[2 * i + 1] = 0  # int16 high 8bit


def UXT2(Td, Tn, Sel):  # Sel control low half or high half
    for i in range(16):
        if Sel == 0:  # low half
            Td[2 * i] = Tn[i]
            Td[2 * i + 1] = 0  # int16 high 8bit
        else:
            Td[2 * i] = Tn[i + 16]
            Td[2 * i + 1] = 0  # int16 high 8bit


def sel(data1, data2, Pg):
    dataout = np.zeros(32)
    for i in range(len(data1)):
        if Pg[i] == 1:
            dataout[i] = data1[i]
        else:
            dataout[i] = data2[i]
    return dataout


def TBL(Td, Tn1, Tn0, Tm):
    Tn = [Tn0, Tn1]
    Tn = np.concatenate(np.array(Tn))
    for i in range(16):
        Td[i * 2] = Tn[int(Tm[i * 2] * 2)]
        Td[i * 2 + 1] = 0


def PZIP(data1, data2, Pg):
    out = np.zeros([32])
    if Pg:
        for i in range(16):
            out[i * 2] = data1[i]
            out[i * 2 + 1] = data2[i]
    else:
        for i in range(16):
            out[i * 2] = data1[i + 16]
            out[i * 2 + 1] = data2[i + 16]

    return out


def shuffle(loop_data, interval_level, Pg, index1, index2):
    T1 = np.zeros([32])
    T2 = np.zeros([32])
    T12 = np.zeros([32])
    T13 = np.zeros([32])
    D0 = np.zeros([32])
    D2 = np.zeros([32])
    P7 = np.zeros([32])
    loop_data1, B2, B3 = PERM(loop_data, interval_level)
    for count in range(32):
        T1[count] = min(loop_data[count], loop_data1[count])
        T2[count] = max(loop_data[count], loop_data1[count])
    data_out = sel(T1, T2, Pg)

    T10, T11 = index1, index2
    D0 = loop_data >= loop_data1
    # D0 = D0[::-1]
    D2 = loop_data <= loop_data1
    # D2 = D2[::-1]
    TBL(T12, T11, T10, B2)
    TBL(T13, T11, T10, B3)
    T10, T11 = T12, T13
    D6 = PZIP(D2, P7, 0)
    D4 = PZIP(D2, P7, 1)
    T12 = sel(T10, index1, D4)
    T13 = sel(T11, index2, D6)
    D6 = PZIP(D0, P7, 0)
    D4 = PZIP(D0, P7, 1)
    T10 = sel(T10, index1, D4)
    T11 = sel(T11, index2, D6)
    D6 = PZIP(Pg, P7, 0)
    D4 = PZIP(Pg, P7, 1)
    index1 = sel(T10, T12, D4)
    index2 = sel(T11, T13, D6)
    return data_out, index1, index2


def bitonic_sort(num_length, data, index_all):
    loop_inter = int(num_length / 32)
    T1 = np.zeros([32])
    T2 = np.zeros([32])
    for j in range(loop_inter):
        loop_data = data[j * 32: j * 32 + 32]
        index = index_all[j * 32:j * 32 + 32]
        index_1, index_2 = split_the_index(index)

        Pg = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 1, Pg, index_1, index_2)

        Pg = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 2, Pg, index_1, index_2)

        Pg = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 1, Pg, index_1, index_2)

        Pg = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 4, Pg, index_1, index_2)

        Pg = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 2, Pg, index_1, index_2)

        Pg = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 1, Pg, index_1, index_2)

        Pg = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 8, Pg, index_1, index_2)

        Pg = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 4, Pg, index_1, index_2)

        Pg = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 2, Pg, index_1, index_2)

        Pg = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        loop_data, index_1, index_2 = shuffle(loop_data, 1, Pg, index_1, index_2)

        Pg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        loop_data, index_1, index_2 = shuffle(loop_data, 16, Pg, index_1, index_2)

        Pg = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        loop_data, index_1, index_2 = shuffle(loop_data, 8, Pg, index_1, index_2)

        Pg = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        loop_data, index_1, index_2 = shuffle(loop_data, 4, Pg, index_1, index_2)

        Pg = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        loop_data, index_1, index_2 = shuffle(loop_data, 2, Pg, index_1, index_2)

        Pg = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        loop_data, index_1, index_2 = shuffle(loop_data, 1, Pg, index_1, index_2)

        data[j * 32: j * 32 + 32] = loop_data
        index_all[j * 32: j * 32 + 32] = merge_the_index(index_1, index_2)

    return data, index_all


def merge_sort(num_length, data, index_all):
    loop_inter = int(num_length / 64)
    data2 = np.zeros([num_length])
    index_all2 = np.zeros([num_length])
    data2[:] = data[:]
    index_all2[:] = index_all[:]

    while loop_inter > 0:
        index = int(loop_inter)
        ld_data = int(num_length / index / 32)
        input_pos = 0
        for k in range(index):
            for j in range(ld_data):
                loop_data = data2[input_pos * 32: input_pos * 32 + 32]
                index_data = index_all2[input_pos * 32: input_pos * 32 + 32]
                if j < int(ld_data / 2):
                    data[j * 32 + int(num_length / index) * k: j * 32 + 32 + int(
                        num_length / index) * k] = loop_data
                    index_all[j * 32 + int(num_length / index) * k: j * 32 + 32 + int(
                        num_length / index) * k] = index_data
                input_pos += 1
                if j >= int(ld_data / 2):
                    loop_data = PERM(loop_data, 32)[0]
                    data[(ld_data - 1 + int(ld_data / 2) - j) * 32 + int(num_length / index) * k: (ld_data - 1 + int(
                        ld_data / 2) - j) * 32 + 32 + int(num_length / index) * k] = loop_data
                    index_all[(ld_data - 1 + int(ld_data / 2) - j) * 32 + int(num_length / index) * k:
                              (ld_data - 1 + int(ld_data / 2) - j) * 32 + 32 + int(
                                  num_length / index) * k] = index_data[::-1]

            ld_data_index = 1
            temp = int(ld_data / 2)
            temp2 = ld_data * 2
            while temp > 0:
                temp2 = int(temp2 / 2)
                for i in range(0, ld_data_index):
                    for j in range(0, temp):
                        temp_index = j * 32 + int(num_length / index) * k + temp2 * i * 32
                        loop_data1 = data[temp_index:temp_index + 32]
                        loop_data2 = data[temp_index + temp * 32:temp_index + temp * 32 + 32]
                        index_data1 = index_all[temp_index:temp_index + 32]
                        index_data2 = index_all[temp_index + temp * 32:temp_index + temp * 32 + 32]
                        T30 = np.zeros([32])
                        T29 = np.zeros([32])
                        for temp_i in range(32):
                            T29[temp_i] = max(loop_data1[temp_i], loop_data2[temp_i])
                            T30[temp_i] = min(loop_data1[temp_i], loop_data2[temp_i])

                        P7 = np.zeros([32])
                        T6, T7 = split_the_index(index_data1)
                        T8, T9 = split_the_index(index_data2)
                        P3 = np.array(loop_data1) >= loop_data2
                        P4 = np.array(loop_data1) < loop_data2
                        D6 = PZIP(P4, P7, 0)
                        D4 = PZIP(P4, P7, 1)
                        T12 = sel(T6, T8, D4)
                        T13 = sel(T7, T9, D6)

                        D6 = PZIP(P3, P7, 0)
                        D4 = PZIP(P3, P7, 1)
                        T10 = sel(T6, T8, D4)
                        T11 = sel(T7, T9, D6)

                        index_all[temp_index:temp_index + 32] = merge_the_index(T10, T11)
                        index_all[temp_index + temp * 32:temp_index + temp * 32 + 32] = merge_the_index(T12, T13)

                        data[temp_index:temp_index + 32] = T29
                        data[temp_index + temp * 32:temp_index + temp * 32 + 32] = T30
                ld_data_index *= 2
                temp = int(temp / 2)

            for j in range(ld_data):
                loop_data = data[j * 32 + int(num_length / index) * k:int(num_length / index) * k + j * 32 + 32]
                index_data = index_all[j * 32 + int(num_length / index) * k:int(num_length / index) * k + j * 32 + 32]
                index_1, index_2 = split_the_index(index_data)
                # interval 16
                # Tx[22]  Px[0]
                Pg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                loop_data, index_1, index_2 = shuffle(loop_data, 16, Pg, index_1, index_2)

                # interval 8
                # Tx[23]  Px[3]
                Pg = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
                loop_data, index_1, index_2 = shuffle(loop_data, 8, Pg, index_1, index_2)

                # interval 4
                # Tx[24]  Px[4]
                Pg = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
                loop_data, index_1, index_2 = shuffle(loop_data, 4, Pg, index_1, index_2)

                # interval 2
                # Tx[25]  Px[5]
                Pg = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                loop_data, index_1, index_2 = shuffle(loop_data, 2, Pg, index_1, index_2)

                # interval 1
                # Tx[26]  Px[6]
                Pg = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                loop_data, index_1, index_2 = shuffle(loop_data, 1, Pg, index_1, index_2)

                data[j * 32 + int(num_length / index) * k:int(num_length / index) * k + j * 32 + 32] = loop_data
                data2[j * 32 + int(num_length / index) * k:j * 32 + int(num_length / index) * k + 32] = loop_data
                index_all[
                j * 32 + int(num_length / index) * k:int(num_length / index) * k + j * 32 + 32] = merge_the_index(
                    index_1, index_2)
                index_all2[
                j * 32 + int(num_length / index) * k:int(num_length / index) * k + j * 32 + 32] = merge_the_index(
                    index_1, index_2)
        loop_inter = int(loop_inter / 2)

    return data2, index_all2


### bitonic_sort 16bit releated function
def bitonic_sort_16bit(num_length, data, index_all):
    lenGlobalSRAM = 256 * 1024  # Global SPRAM: 256KB
    globalSRAM = np.zeros([lenGlobalSRAM])
    globalSRAMIndex = np.zeros([lenGlobalSRAM])

    globalSRAM[0:num_length] = data[0:num_length]  # instead localsram not dma

    globalSRAMIndex[0:num_length] = index_all[0:num_length]

    # Current position of input in globalSRAM
    input_pos = 0

    # TEC_num x 32 x 16bytes tensor vector register (Tn, Tm, Td)
    Tx = np.zeros([32, 16])

    Px = np.ones([8, 32], dtype=bool)
    # interval 1
    # Tx[26]
    Tx[26][0] = 1
    Tx[26][1] = 0
    Tx[26][2] = 3
    Tx[26][3] = 2
    Tx[26][4] = 5
    Tx[26][5] = 4
    Tx[26][6] = 7
    Tx[26][7] = 6
    Tx[26][8] = 9
    Tx[26][9] = 8
    Tx[26][10] = 11
    Tx[26][11] = 10
    Tx[26][12] = 13
    Tx[26][13] = 12
    Tx[26][14] = 15
    Tx[26][15] = 14

    # interval 2
    Tx[25][0] = 2
    Tx[25][1] = 3
    Tx[25][2] = 0
    Tx[25][3] = 1
    Tx[25][4] = 6
    Tx[25][5] = 7
    Tx[25][6] = 4
    Tx[25][7] = 5
    Tx[25][8] = 10
    Tx[25][9] = 11
    Tx[25][10] = 8
    Tx[25][11] = 9
    Tx[25][12] = 14
    Tx[25][13] = 15
    Tx[25][14] = 12
    Tx[25][15] = 13

    # interval 4
    Tx[24][0] = 4
    Tx[24][1] = 5
    Tx[24][2] = 6
    Tx[24][3] = 7
    Tx[24][4] = 0
    Tx[24][5] = 1
    Tx[24][6] = 2
    Tx[24][7] = 3
    Tx[24][8] = 12
    Tx[24][9] = 13
    Tx[24][10] = 14
    Tx[24][11] = 15
    Tx[24][12] = 8
    Tx[24][13] = 9
    Tx[24][14] = 10
    Tx[24][15] = 11

    # interval 8
    # Tx[23]  Px[3]
    Tx[23][0] = 8
    Tx[23][1] = 9
    Tx[23][2] = 10
    Tx[23][3] = 11
    Tx[23][4] = 12
    Tx[23][5] = 13
    Tx[23][6] = 14
    Tx[23][7] = 15
    Tx[23][8] = 0
    Tx[23][9] = 1
    Tx[23][10] = 2
    Tx[23][11] = 3
    Tx[23][12] = 4
    Tx[23][13] = 5
    Tx[23][14] = 6
    Tx[23][15] = 7

    # reverse
    Tx[31][0] = 15
    Tx[31][1] = 14
    Tx[31][2] = 13
    Tx[31][3] = 12
    Tx[31][4] = 11
    Tx[31][5] = 10
    Tx[31][6] = 9
    Tx[31][7] = 8
    Tx[31][8] = 7
    Tx[31][9] = 6
    Tx[31][10] = 5
    Tx[31][11] = 4
    Tx[31][12] = 3
    Tx[31][13] = 2
    Tx[31][14] = 1
    Tx[31][15] = 0

    # TEC_num x 8 x 16bits predication register (Pg, Pn, Pd)
    Px = np.ones([8, 16], dtype=bool)

    loop_inter = np.uint32(num_length / 16)
    for i in range(loop_inter):
        TLDS_global(globalSRAM, Tx[0], input_pos, 1, Px[0])
        # Tx[0] = [40, 17, 49,  0, 31,  6, 13, 42, 28,  3, 43, 15,  9,  4, 24, 16]
        TLDS_global_index(globalSRAMIndex, Tx[15], input_pos, 1, Px[0])

        Px[1] = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]  # 0x6666 0x14141414
        shuffle_16bit(Tx[1], Tx[0], Tx[26], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 1 Tx[26]

        Px[1] = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  # 0x3c3c 0x05500550
        shuffle_16bit(Tx[1], Tx[0], Tx[25], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 2 Tx[25]

        Px[1] = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  # 0x5a5a 0x11441144
        shuffle_16bit(Tx[1], Tx[0], Tx[26], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 1 Tx[26]

        Px[1] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 0x0ff0 0x00555500
        shuffle_16bit(Tx[1], Tx[0], Tx[24], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 4 Tx[24]

        Px[1] = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  # 0x33cc 0x05055050
        shuffle_16bit(Tx[1], Tx[0], Tx[25], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 2 Tx[25]

        Px[1] = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  # 0x55aa 0x11114444
        shuffle_16bit(Tx[1], Tx[0], Tx[26], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 1 Tx[26]

        ###################
        Px[1] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 0xff00 0x00005555
        shuffle_16bit(Tx[1], Tx[0], Tx[23], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 8 Tx[23]

        Px[1] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  # 0xf0f0 0x00550055
        shuffle_16bit(Tx[1], Tx[0], Tx[24], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 4 Tx[24]

        Px[1] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  # 0xcccc 0x05050505
        shuffle_16bit(Tx[1], Tx[0], Tx[25], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 2 Tx[25]

        Px[1] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 0xaaaa 0x11111111
        shuffle_16bit(Tx[1], Tx[0], Tx[26], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5], Tx[6],
                      Tx[7])  # interval 1 Tx[26]

        # if i == 0 or i == 8 or i == 16 or i == 24:
        #     print("debug")

        for ii in range(15):
            if Tx[0][ii] > Tx[0][ii + 1]:
                print("error")
        TSTS_global(globalSRAM, Tx[0], input_pos, 1, Px[0])
        TSTS_global_index(globalSRAMIndex, Tx[15], input_pos, 1, Px[0])
        input_pos = input_pos + 16

    globalSRAMHalf = 128 * 1024
    loop_inter = int(num_length / 16 / 2)  # how much groups can be slices,and per group has 32 bytes.
    while int(loop_inter) > 0:
        index = int(loop_inter)
        ld_data = int(num_length / index / 16)
        input_pos = 0

        for k in range(index):
            Px[0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

            for j in range(ld_data):
                TLDS_global(globalSRAM, Tx[0], input_pos, 1, Px[0])
                TLDS_global_index(globalSRAMIndex, Tx[15], input_pos, 1, Px[0])

                if j < int(ld_data / 2):
                    TSTS_global(globalSRAM, Tx[0], globalSRAMHalf + 16 * j + int(num_length / index) * k, 1, Px[0])
                    TSTS_global_index(globalSRAMIndex, Tx[15], globalSRAMHalf + 16 * j + int(num_length / index) * k, 1,
                                      Px[0])
                input_pos += 16
                if j >= int(ld_data / 2):  # high half reverse
                    PERM_16BIT(Tx[30], Tx[0], Tx[31])
                    TSTS_global(globalSRAM, Tx[30], globalSRAMHalf + 16 * (ld_data - 1 - (j - int(ld_data / 2))) + int(
                        num_length / index) * k, 1, Px[0])
                    PERM_16BIT(Tx[30], Tx[15], Tx[31])
                    TSTS_global_index(globalSRAMIndex, Tx[30],
                                      globalSRAMHalf + 16 * (ld_data - 1 - (j - int(ld_data / 2))) + int(
                                          num_length / index) * k, 1, Px[0])

            ld_data_index = 1
            temp = int(ld_data / 2)
            temp2 = int(ld_data * 2)
            while temp > 0:
                temp2 = int(temp2 / 2)
                for i in range(0, ld_data_index):
                    for j in range(0, temp):
                        TLDS_global(globalSRAM, Tx[0],
                                    globalSRAMHalf + 16 * j + int(num_length / index) * k + temp2 * i * 16,
                                    1, Px[0])
                        TLDS_global(globalSRAM, Tx[1],
                                    globalSRAMHalf + 16 * (j + temp) + int(num_length / index) * k + temp2 * i * 16,
                                    1, Px[0])
                        MIN(Tx[29], Tx[0], Tx[1], Px[0])
                        MAX(Tx[30], Tx[0], Tx[1], Px[0])
                        TSTS_global(globalSRAM, Tx[30],
                                    globalSRAMHalf + 16 * j + int(num_length / index) * k + temp2 * i * 16,
                                    1, Px[7])
                        TSTS_global(globalSRAM, Tx[29],
                                    globalSRAMHalf + 16 * (j + temp) + int(num_length / index) * k + temp2 * i * 16,
                                    1, Px[0])

                        TGE(Tx[0], Tx[1], Px[3])
                        # TLE(Tx[0], Tx[1], Px[4]) #BUG
                        TINV(Px[4], Px[3])
                        TLDS_global_index(globalSRAMIndex, Tx[0],
                                          globalSRAMHalf + 16 * j + int(num_length / index) * k + temp2 * i * 16, 1,
                                          Px[0])
                        TLDS_global_index(globalSRAMIndex, Tx[1], globalSRAMHalf + 16 * (j + temp) + int(
                            num_length / index) * k + temp2 * i * 16, 1, Px[0])
                        TSELH(Tx[5], Tx[0], Tx[1], Px[3])
                        TSELH(Tx[6], Tx[0], Tx[1], Px[4])
                        TSTS_global_index(globalSRAMIndex, Tx[5],
                                          globalSRAMHalf + 16 * j + int(num_length / index) * k + temp2 * i * 16, 1,
                                          Px[0])
                        TSTS_global_index(globalSRAMIndex, Tx[6], globalSRAMHalf + 16 * (j + temp) + int(
                            num_length / index) * k + temp2 * i * 16, 1, Px[0])

                ld_data_index = ld_data_index * 2
                temp = int(temp / 2)

            for j in range(ld_data):
                TLDS_global(globalSRAM, Tx[0], globalSRAMHalf + 16 * j + int(num_length / index) * k, 1, Px[0])
                TLDS_global_index(globalSRAMIndex, Tx[15], globalSRAMHalf + 16 * j + int(num_length / index) * k, 1,
                                  Px[0])

                Px[1] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 0x00ff 0x55550000
                shuffle_16bit(Tx[1], Tx[0], Tx[23], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5],
                              Tx[6],
                              Tx[7])  # interval8 Tx[23]

                Px[1] = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]  # 0x0f0f 0x55005500
                shuffle_16bit(Tx[1], Tx[0], Tx[24], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5],
                              Tx[6],
                              Tx[7])  # interval4 Tx[24]

                Px[1] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  # 0x3333 0x50505050
                shuffle_16bit(Tx[1], Tx[0], Tx[25], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5],
                              Tx[6],
                              Tx[7])  # interval2 Tx[25]

                Px[1] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0x5555 0x44444444
                shuffle_16bit(Tx[1], Tx[0], Tx[26], Px[0], Px[1], Tx[2], Tx[3], Px[3], Px[4], Tx[15], Tx[4], Tx[5],
                              Tx[6],
                              Tx[7])  # interval1 Tx[26]

                TSTS_global(globalSRAM, Tx[0], 16 * j + int(num_length / index) * k, 1, Px[0])
                TSTS_global_index(globalSRAMIndex, Tx[15], 16 * j + int(num_length / index) * k, 1, Px[0])

        loop_inter = int(loop_inter / 2)

    return globalSRAM, globalSRAMIndex


# Tensor stride load from globalSRAM with predication
def TLDS_global(globalSRAM, Td, Rn, imm8, Pg):
    for i in range(16):
        if Pg[i] == True:
            Td[i] = globalSRAM[Rn + i * imm8]
        else:
            Td[i] = 0


def TSTS_global(globalSRAM, Td, Rn, imm8, Pg):
    for i in range(16):
        if Pg[i] == True:
            globalSRAM[Rn + i * imm8] = Td[i]


def TLDS_global_index(globalSRAMIndex, Td, Rn, imm8, Pg):
    for i in range(16):
        if Pg[i] == True:
            Td[i] = globalSRAMIndex[Rn + i * imm8]
        else:
            Td[i] = 0


def TSTS_global_index(globalSRAMIndex, Td, Rn, imm8, Pg):
    for i in range(16):
        if Pg[i] == True:
            globalSRAMIndex[Rn + i * imm8] = Td[i]


# Minimum vectors (predicated)
def MIN(Td, Tn, Tm, Pg):
    for i in range(16):
        if Pg[i] == True:
            if Tn[i] <= Tm[i]:
                Td[i] = Tn[i]
            else:
                Td[i] = Tm[i]
        else:
            Td[i] = 0


# Maximum vectors (predicated)
def MAX(Td, Tn, Tm, Pg):
    for i in range(16):
        if Pg[i] == True:
            if Tn[i] >= Tm[i]:
                Td[i] = Tn[i]
            else:
                Td[i] = Tm[i]
        else:
            Td[i] = 0


# index ->Tm
def PERM_16BIT(Td, Tn, Tm):
    for i in range(16):
        j = int(Tm[i])
        Td[i] = Tn[j]


# select element
def SEL(Td, Tn, Tm, Pg):
    for i in range(16):
        if Pg[i] == True:
            Td[i] = Tn[i]
        else:
            Td[i] = Tm[i]


def TGE(Tn, Tm, Ptmp):
    for i in range(16):
        if Tn[i] >= Tm[i]:
            Ptmp[i] = True
        else:
            Ptmp[i] = False


def TINV(Pg1, Pg0):
    for i in range(16):
        if Pg0[i] == True:
            Pg1[i] = False
        else:
            Pg1[i] = True


def TLE(Tn, Tm, Ptmp):
    for i in range(16):
        if Tn[i] <= Tm[i]:
            Ptmp[i] = True
        else:
            Ptmp[i] = False


def TSELH(Td, Tn, Tm, Pg):
    for i in range(16):
        if Pg[i] == True:
            Td[i] = Tn[i]
        else:
            Td[i] = Tm[i]


def shuffle_16bit(Td, Tn, Tm, Pg1, Pg2, Temp1, Temp2, Pg3, Pg4, Tindex, Temp3, Temp4, Temp5, Temp6):
    PERM_16BIT(Td, Tn, Tm)
    MAX(Temp1, Td, Tn, Pg1)
    MIN(Temp2, Td, Tn, Pg1)

    TGE(Td, Tn, Pg3)
    TLE(Td, Tn, Pg4)
    PERM_16BIT(Temp3, Tindex, Tm)
    TSELH(Temp4, Temp3, Tindex, Pg3)
    TSELH(Temp5, Temp3, Tindex, Pg4)
    TSELH(Temp6, Temp5, Temp4, Pg2)

    SEL(Td, Temp2, Temp1, Pg2)

    for i in range(16):
        Tn[i] = Td[i]
        Tindex[i] = Temp6[i]
