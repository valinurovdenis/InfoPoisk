type_1 = 0x10000000
type_2 = 0x20000000
type_3 = 0x30000000
type_4 = 0x40000000
type_5 = 0x50000000
type_6 = 0x60000000
type_7 = 0x70000000
type_8 = 0x80000000
type_9 = 0x90000000

powers_of_2 = [2 ** i for i in range(29)]
bytes_border = [16777216, 65536, 256, 1]
bits_in_byte = 256

encode_struct = [[28, type_1, 1],
                 [14, type_2, 2],
                 [9, type_3, 3],
                 [7, type_4, 4],
                 [5, type_5, 5],
                 [4, type_6, 4],
                 [3, type_7, 9],
                 [2, type_8, 14],
                 [1, type_9, 28]]


def encode(array):
    global encode_struct
    ret = bytearray()

    to_encode = []
    prev = 0
    for cur in array:
        dif = cur - prev
        prev = cur
        to_encode.append(dif)

    pos = 0
    while pos < len(to_encode):
        for type_info in encode_struct:
            cnt, code, offset = type_info
            if pos + cnt <= len(to_encode) and max(to_encode[pos:pos + cnt]) < powers_of_2[offset]:
                to_append = 0
                for i, num in enumerate(to_encode[pos:pos + cnt]):
                    to_append |= (num << (offset * i))

                to_append |= code
                ret.extend([(to_append / border) % bits_in_byte for border in bytes_border])
                pos += cnt
                break
                 
    return ret



def decode(array):
    global encode_struct
    ret = []

    pos = 0
    while pos < len(array):
        encoded_numbers = 0
        for i, border in enumerate(bytes_border):
            encoded_numbers += array[pos + i] * border  

        code = encoded_numbers & 0xf0000000
        data = encoded_numbers & 0x0fffffff
        cnt, code, offset = encode_struct[(code >> 28) - 1]
        for i in xrange(cnt):
            ret.append(data % powers_of_2[offset])
            data >>= offset

        pos += 4
    
    for i in xrange(len(ret) - 1):
        ret[i + 1] += ret[i]

    return ret
