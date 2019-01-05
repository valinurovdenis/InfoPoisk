def encode(array):
    prev = 0
    arr = bytearray()
    for cur in array:
        dif = cur - prev
        prev = cur
        buf = []
        while dif >= 128:
            buf.append(dif % 128)
            dif /= 128
        buf.append(dif)
        buf[0] += 128

        for ind in range(len(buf))[::-1]:
            arr.append(buf[ind])

    return arr


def decode(array):
    num, res = 0, 0
    result = []
    for cur in array:
        if cur < 128:
            res = 128 * res + cur
        else:
            res = 128 * res + (cur - 128)
            num += res
            result.append(num)
            res = 0

    return result
