# -*- coding: utf-8 -*-
import sys
import mmh3
from array import array
import parsetree
import coder
import cPickle as pickle


def tree_search(request, omega_id):
    decoder = coder.Coder()
    result = []
    goto_docid = 1

    while goto_docid is not None and goto_docid <= omega_id:
        request.goto(goto_docid)
        (docid, goto_docid) = request.evaluate()
        if docid is not None and docid <= omega_id:
            result.append(docid)

    return result


if __name__ == '__main__':
    paths = pickle.load(open('paths.pkl', 'r'))

    for request in sys.stdin:
        request_tree = parsetree.parse_query(request.decode('utf8').lower(), len(paths)+1)
        result = tree_search(request_tree, len(paths))
        sys.stdout.write(request.rstrip() + '\n')
        sys.stdout.write(str(len(result)) + '\n')
        for key in result:
            sys.stdout.write(paths[key - 1] + '\n')
