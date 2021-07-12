#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import os
import inspect


def trace_function(function, log_calls, log_results, label=''):
    def wrapper(*args, **kwargs):
        kwarg_strs = ['{}={}'.format(k, v) for (k, v) in kwargs]
        arg_str = ', '.join([str(a) for a in args] + kwarg_strs)
        call_str = '{}({})'.format(function.__name__, arg_str)

        # Perform the call itself, logging before, after, and anything thrown.
        try:
            if log_calls:
                print('{}: Calling {}'.format(label, call_str))
            res = function(*args, **kwargs)
            if log_results:
                print('{}: {} -> {}'.format(label, call_str, res))
            return res
        except Exception as ex:
            if log_results:
                print('{}: {} raised {}'.format(label, call_str, type(ex)))
            raise ex

    return wrapper


def trace_object(obj, log_calls, log_results, label=''):
    for name, member in inspect.getmembers(obj):
        if inspect.ismethod(member):
            # Skip meta-functions, decorate everything else
            if not member.__name__.startswith('__'):
                setattr(obj, name, trace_function(member, log_calls,
                                                  log_results, label))
    return obj
