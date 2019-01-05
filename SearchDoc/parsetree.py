import re
import mmh3
from array import array
import coder

SPLIT_RGX = re.compile(r'\w+|[\(\)&\|!]', re.U)

bin_dictionary = open('dictionary.bin', 'rb')
bin_index = open('index.bin', 'rb')
omega_id = -1

def term_search(hash_key):
    buf = array('l')
    bin_dictionary.seek(0)
    buf.fromfile(bin_dictionary, 1)
    module = buf.pop()
    dict_key = hash_key % module
    long_size = 8
    bin_dictionary.seek(long_size * (2 * dict_key + 1))
    buf.fromfile(bin_dictionary, 2)
    count, offset = buf.pop(), buf.pop()

    bin_dictionary.seek(long_size * offset)
    buf.fromfile(bin_dictionary, long_size * count * 3)

    l = -1
    r = count

    while l + 1 < r:
        mid = (l + r) / 2
        if buf[3 * mid] < hash_key:
            l = mid
        else:
            r = mid

    ret_offset, ret_count = None, None

    if buf[3 * r] == hash_key:
        ret_offset = buf[3 * r + 1]
        ret_count = buf[3 * r + 2]

    return (ret_offset, ret_count)


class TreeNode:
    def __init__(self, val, bracket=False, term=False, op=False):
        self.val = val
        self.is_operator = op
        self.is_bracket = bracket
        self.is_term = term
        self.omega_id = omega_id
        if self.is_operator:
            self.prio = operator_prio(self.val)


class TreeNodeOper(TreeNode):
    def __init__(self, val):
        TreeNode.__init__(self, val, op=True)
        self.current_docid = -1


    def goto(self, docid):
        if self.val == '&' or self.val == '|':
            self.left.goto(docid)
            self.right.goto(docid)
        else:
            if self.current_docid < docid:
                self.right.goto(docid)
                r = self.right.evaluate()
                while docid < self.omega_id and r[0] == docid:
                    docid += 1
                    if r[1] != docid:
                        break
                    self.right.goto(docid)
                    r = self.right.evaluate()
                self.current_docid = docid


    def evaluate(self):
        if self.val == '&':
            l = self.left.evaluate()
            r = self.right.evaluate()

            if l[0] == r[0]:
                return (l[0], max(l[1], r[1]))
            else:
                if l[0] is None and r[0] is None:
                    return (None, max(l[1], r[1]))
                if l[0] is None:
                    return (None, l[1])
                if r[0] is None:
                    return (None, r[1])
                return (None, max(l[0], r[0]))
        elif self.val == '|':
            l = self.left.evaluate()
            r = self.right.evaluate()

            if l[0] is None or r[0] is None:
                return (max(l[0], r[0]), min(l[1], r[1]))

            if l[0] == r[0]:
                return (l[0], min(l[1], r[1]))
            elif l[0] < r[0]:
                return (l[0], min(l[1], r[0]))
            else:
                return (r[0], min(r[1], l[0]))
        else:
            if self.current_docid < self.omega_id:
                return (self.current_docid, self.current_docid+1)
            else:
                return (None, self.current_docid+1)


class TreeNodeTerm(TreeNode):
    def __init__(self, val):
        TreeNode.__init__(self, val, term=True)
        decoder = coder.Coder()

        hash_val = mmh3.hash64(val.encode('utf8'))[0]
        offset, count = term_search(hash_val)
        if offset is None or count is None:
            self.docs = []
            self.pos = 0
        else:
            bin_index.seek(offset)
            byte_str = bytearray()
            buf = bin_index.read(count)
            byte_str.extend(buf)
            self.docs = decoder.decode(byte_str)
            self.pos = 0


    def goto(self, docid):
        if docid >= self.omega_id or self.pos >= len(self.docs):
            return

        if abs(self.docs[self.pos] - docid) < 20:
            while self.pos < len(self.docs) and self.docs[self.pos] < docid:
                self.pos += 1
        else:
            l = self.pos - 1
            r = len(self.docs)

            while l + 1 < r:
                mid = (l + r) / 2
                if self.docs[mid] < docid:
                    l = mid
                else:
                    r = mid

            self.pos = r


    def evaluate(self):
        if self.pos < len(self.docs):
            if self.pos+1 < len(self.docs):
                return (self.docs[self.pos], self.docs[self.pos+1])
            else:
                return (self.docs[self.pos], self.omega_id)
        else:
            return (None, self.omega_id)


def operator_prio(c):
    if c == '|':
        return 0
    elif c == '&':
        return 1
    elif c == '!':
        return 2

    return None


def is_operator(c):
    return (c == '|' or c == '&' or c == '!')

def is_bracket(c):
    return (c == '(' or c == ')')

def tokenize(q):
    tokenized = []
    for token in re.findall(SPLIT_RGX, q):
        if is_bracket(token):
            tokenized.append(TreeNode(token, bracket=True))
        elif is_operator(token):
            tokenized.append(TreeNodeOper(token))
        else:
            tokenized.append(TreeNodeTerm(token))

    return tokenized


def build_tree(tokens):
    min_depth = float('inf')
    min_prio = float('inf')
    depth = 0
    ind = -1

    for i, token in enumerate(tokens):
        if token.is_operator:
            if depth < min_depth or (depth == min_depth and token.prio < min_prio):
                ind = i
                min_prio = token.prio
                min_depth = depth
        elif token.is_bracket:
            if token.val == '(':
                depth += 1
            else:
                depth -= 1

    if ind != -1:
        result = tokens[ind]
        result.left = build_tree(tokens[:ind])
        result.right = build_tree(tokens[ind+1:])
        if result.val == '!' and result.right.val == '!':
            return result.right.right
        return result
    else:
        for token in tokens:
            if token.is_term:
                return token

        return None


def parse_query(query, omega):
    global omega_id
    omega_id = omega
    return build_tree(tokenize(query))
